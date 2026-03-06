//! Decode worker: receives the KV cache from a prefill worker and runs the
//! autoregressive token generation loop.

use anyhow::Result;
use nexar::{NexarClient, Rank};
use std::sync::Arc;

use super::protocol::{DecodeRequest, DecodeStepFn, DecodedToken, tags};
use crate::distributed::inference::transport;

/// Decode worker: receives the KV cache from a prefill worker and runs the
/// autoregressive token generation loop.
///
/// # Decode step function contract
///
/// `decode_step_fn` receives:
/// - `kv_cache_bytes`: the full serialised KV cache for the current request
/// - `last_token_id`: the most recently generated token (or the last prompt
///   token for the very first decode step)
/// - `position`: current sequence position (prompt length + tokens generated so far)
///
/// It must return `(next_token_id, updated_kv_cache_bytes)`.  The updated KV
/// cache is passed back on the next call so the function does not need to
/// maintain internal state between calls (pure functional interface).
pub struct DecodeWorker {
    client: Arc<NexarClient>,
    rank: Rank,
    router_rank: Rank,
    /// Ranks of all configured prefill workers that may send KV caches.
    prefill_workers: Vec<Rank>,
    max_kv_transfer_bytes: usize,
    /// `(next_token_id, updated_kv) = decode_step_fn(kv_cache, last_token_id, position)`
    decode_step_fn: DecodeStepFn,
}

impl DecodeWorker {
    /// Create a decode worker.
    ///
    /// `prefill_workers` lists all prefill worker ranks that may send KV caches
    /// to this decode worker. The worker tries each rank in order when waiting
    /// for an incoming KV cache transfer.
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        router_rank: Rank,
        prefill_workers: Vec<Rank>,
        max_kv_transfer_bytes: usize,
        decode_step_fn: DecodeStepFn,
    ) -> Self {
        Self {
            client,
            rank,
            router_rank,
            prefill_workers,
            max_kv_transfer_bytes,
            decode_step_fn,
        }
    }

    /// Run the decode worker's event loop.
    ///
    /// The loop:
    /// 1. Receives a KV cache (length header then payload) from a prefill worker.
    /// 2. Sends `KV_CACHE_ACK` to the prefill worker.
    /// 3. Waits for a `DecodeRequest` from the router.
    /// 4. Runs the autoregressive decode loop, streaming tokens back to the router.
    /// 5. Sends `DECODE_DONE` when generation is complete.
    pub async fn run_loop(&self) -> Result<()> {
        tracing::info!(rank = self.rank, "Decode worker loop starting");

        loop {
            // Step 1a: receive KV cache length from whichever prefill worker is
            // sending. We try each configured prefill worker rank in turn and
            // accept from the first one that has data.
            let mut kv_len_buf = [0u8; 8];
            let mut kv_source_rank: Rank = 0;
            let mut received = false;

            for &prefill_rank in &self.prefill_workers {
                match transport::recv_bytes(
                    &self.client,
                    &mut kv_len_buf,
                    prefill_rank,
                    tags::KV_CACHE,
                )
                .await
                {
                    Ok(()) => {
                        kv_source_rank = prefill_rank;
                        received = true;
                        break;
                    }
                    Err(_) => continue,
                }
            }

            if !received {
                tracing::warn!(
                    rank = self.rank,
                    "No KV cache received from any prefill worker (shutdown?)"
                );
                break;
            }

            let kv_len = u64::from_le_bytes(kv_len_buf) as usize;

            if kv_len > self.max_kv_transfer_bytes {
                tracing::error!(
                    rank = self.rank,
                    kv_bytes = kv_len,
                    limit = self.max_kv_transfer_bytes,
                    "KV cache exceeds limit; skipping request"
                );
                continue;
            }

            // Step 1b: receive the actual KV cache payload.
            let mut kv_cache = vec![0u8; kv_len];
            transport::recv_bytes(&self.client, &mut kv_cache, kv_source_rank, tags::KV_CACHE)
                .await?;

            // Step 2: acknowledge receipt so the prefill worker can tell the
            // router that it is done.
            let ack = [0u8; 8];
            transport::send_bytes(&self.client, &ack, kv_source_rank, tags::KV_CACHE_ACK).await?;

            // Step 3: wait for the decode request from the router.
            let mut decode_req_buf = [0u8; 16];
            transport::recv_bytes(
                &self.client,
                &mut decode_req_buf,
                self.router_rank,
                tags::DECODE_REQUEST,
            )
            .await?;
            let decode_req = DecodeRequest::from_bytes(&decode_req_buf);

            tracing::debug!(
                rank = self.rank,
                request_id = decode_req.request_id,
                max_new_tokens = decode_req.max_new_tokens,
                "Starting decode loop"
            );

            // Step 4: autoregressive generation.
            let mut last_token: i64 = 0;
            let mut position: u32 = 0;
            let mut tokens_generated: u32 = 0;

            // Extract the starting position from the KV cache header (first 4
            // bytes, convention agreed between prefill_fn and decode_step_fn).
            if kv_cache.len() >= 4 {
                position = u32::from_le_bytes(kv_cache[0..4].try_into().unwrap());
            }

            loop {
                if tokens_generated >= decode_req.max_new_tokens {
                    break;
                }

                let (next_token, updated_kv) =
                    (self.decode_step_fn)(&kv_cache, last_token, position);

                kv_cache = updated_kv;
                last_token = next_token;
                position += 1;
                tokens_generated += 1;

                // Stream the token back to the router.
                let tok = DecodedToken {
                    request_id: decode_req.request_id,
                    token_id: next_token,
                };
                transport::send_bytes(
                    &self.client,
                    &tok.to_bytes(),
                    self.router_rank,
                    tags::DECODE_TOKEN,
                )
                .await?;

                // EOS token convention: i64::MIN signals end of sequence.
                if next_token == i64::MIN {
                    tracing::debug!(
                        rank = self.rank,
                        request_id = decode_req.request_id,
                        tokens_generated,
                        "EOS reached"
                    );
                    break;
                }
            }

            // Step 5: send DECODE_DONE so the router can close the response.
            let done_payload = decode_req.request_id.to_le_bytes();
            transport::send_bytes(
                &self.client,
                &done_payload,
                self.router_rank,
                tags::DECODE_DONE,
            )
            .await?;

            tracing::debug!(
                rank = self.rank,
                request_id = decode_req.request_id,
                tokens_generated,
                "Decode complete"
            );
        }

        tracing::info!(rank = self.rank, "Decode worker loop ended");
        Ok(())
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }
}
