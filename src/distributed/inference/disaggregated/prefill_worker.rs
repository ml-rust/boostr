//! Prefill worker: runs the prompt forward pass, serialises the resulting KV
//! cache, and pushes it to the decode worker chosen by the router.

use anyhow::Result;
use nexar::{NexarClient, Rank};
use std::sync::Arc;

use super::protocol::{PrefillDone, PrefillFn, PrefillRequest, tags};
use crate::distributed::inference::transport;

/// Prefill worker: runs the prompt forward pass, serialises the resulting KV
/// cache, and pushes it to the decode worker chosen by the router.
///
/// # Forward function contract
///
/// `prefill_fn` receives:
/// - `token_ids_bytes`: raw prompt token IDs as little-endian `i64` bytes
/// - `seq_len`: number of tokens
///
/// It must return `(activation_bytes, kv_cache_bytes)`:
/// - `activation_bytes`: the final hidden state after all assigned layers
/// - `kv_cache_bytes`: the full serialised KV cache for this prompt (all layers)
pub struct PrefillWorker {
    client: Arc<NexarClient>,
    rank: Rank,
    router_rank: Rank,
    /// Upper bound on KV bytes allowed per request (matches `DisaggConfig`).
    max_kv_transfer_bytes: usize,
    /// `(activation_bytes, kv_cache_bytes) = prefill_fn(token_ids_bytes, seq_len)`
    prefill_fn: PrefillFn,
}

impl PrefillWorker {
    /// Create a prefill worker.
    ///
    /// `prefill_fn` is provided by the caller (who has access to the model and
    /// runtime) and performs the actual forward computation.
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        router_rank: Rank,
        max_kv_transfer_bytes: usize,
        prefill_fn: PrefillFn,
    ) -> Self {
        Self {
            client,
            rank,
            router_rank,
            max_kv_transfer_bytes,
            prefill_fn,
        }
    }

    /// Run the prefill worker's event loop.
    ///
    /// The loop:
    /// 1. Receives the `PrefillRequest` descriptor (contains seq_len and the
    ///    target decode rank).
    /// 2. Receives token ID bytes from the router.
    /// 3. Calls `prefill_fn` to run the forward pass and obtain the KV cache.
    /// 4. Sends the KV cache to the decode worker, waits for `KV_CACHE_ACK`.
    /// 5. Sends `PrefillDone` to the router.
    pub async fn run_loop(&self) -> Result<()> {
        tracing::info!(rank = self.rank, "Prefill worker loop starting");

        loop {
            // Step 1: receive the prefill request descriptor.
            let mut req_buf = [0u8; 16];
            match transport::recv_bytes(
                &self.client,
                &mut req_buf,
                self.router_rank,
                tags::PREFILL_REQUEST,
            )
            .await
            {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!(rank = self.rank, "Prefill recv error (shutdown?): {}", e);
                    break;
                }
            }

            let req = PrefillRequest::from_bytes(&req_buf);
            let token_bytes = req.seq_len as usize * std::mem::size_of::<i64>();

            tracing::debug!(
                rank = self.rank,
                request_id = req.request_id,
                seq_len = req.seq_len,
                decode_rank = req.decode_rank,
                "Received prefill request"
            );

            // Step 2: receive the actual token ID bytes.
            let mut token_buf = vec![0u8; token_bytes];
            transport::recv_bytes(
                &self.client,
                &mut token_buf,
                self.router_rank,
                transport::tags::ACTIVATION,
            )
            .await?;

            // Step 3: run prefill forward pass.
            let (_activation, kv_cache) = (self.prefill_fn)(&token_buf, req.seq_len as usize);

            if kv_cache.len() > self.max_kv_transfer_bytes {
                tracing::error!(
                    rank = self.rank,
                    request_id = req.request_id,
                    kv_bytes = kv_cache.len(),
                    limit = self.max_kv_transfer_bytes,
                    "KV cache exceeds transfer limit; dropping request"
                );
                // Notify router with kv_bytes = 0 so it doesn't hang.
                let done = PrefillDone {
                    request_id: req.request_id,
                    kv_bytes: 0,
                };
                let _ = transport::send_bytes(
                    &self.client,
                    &done.to_bytes(),
                    self.router_rank,
                    tags::PREFILL_DONE,
                )
                .await;
                continue;
            }

            let kv_bytes_len = kv_cache.len() as u64;
            let decode_rank = req.decode_rank as Rank;

            // Step 4: send KV cache length header (u64 LE) then the payload.
            let kv_len_bytes = kv_bytes_len.to_le_bytes();
            transport::send_bytes(&self.client, &kv_len_bytes, decode_rank, tags::KV_CACHE).await?;
            transport::send_bytes(&self.client, &kv_cache, decode_rank, tags::KV_CACHE).await?;

            // Wait for the decode worker to acknowledge receipt.
            let mut ack_buf = [0u8; 8];
            transport::recv_bytes(&self.client, &mut ack_buf, decode_rank, tags::KV_CACHE_ACK)
                .await?;

            // Step 5: tell the router that prefill is done.
            let done = PrefillDone {
                request_id: req.request_id,
                kv_bytes: kv_bytes_len,
            };
            transport::send_bytes(
                &self.client,
                &done.to_bytes(),
                self.router_rank,
                tags::PREFILL_DONE,
            )
            .await?;

            tracing::debug!(
                rank = self.rank,
                request_id = req.request_id,
                kv_bytes = kv_bytes_len,
                "Prefill done; KV cache transferred"
            );
        }

        tracing::info!(rank = self.rank, "Prefill worker loop ended");
        Ok(())
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }
}
