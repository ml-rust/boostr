//! Disaggregated inference router.
//!
//! Assigns requests to prefill workers (least-loaded) and routes completed
//! prefills to decode workers (cache-aware affinity or round-robin fallback).

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::protocol::{
    DecodeRequest, DecodedToken, DisaggConfig, PrefillDone, PrefillRequest, tags,
};
use crate::distributed::inference::transport;

/// Per-prefill-worker load counter (number of requests currently being prefilled).
struct PrefillLoad {
    rank: Rank,
    in_flight: AtomicU64,
}

/// Router: assigns requests to prefill workers and routes completed prefills to
/// decode workers.
///
/// Routing policy:
/// - **Prefill**: round-robin over all prefill workers, breaking ties by
///   choosing the one with the fewest in-flight requests (least loaded).
/// - **Decode**: cache-aware — if a decode worker already holds warm KV state
///   for related prior requests from the same logical session it is preferred.
///   Falls back to round-robin over decode workers when no warm affinity exists.
pub struct DisaggRouter {
    client: Arc<NexarClient>,
    config: DisaggConfig,
    /// Monotonically increasing request ID counter.
    next_request_id: AtomicU64,
    /// Per-prefill-worker load tracking.
    prefill_loads: Vec<Arc<PrefillLoad>>,
    /// Round-robin cursor for decode workers (used when no warm KV affinity).
    decode_cursor: AtomicU64,
    /// Cache-affinity map: session_key → decode_rank.
    ///
    /// When a request belongs to a known session the router prefers the decode
    /// worker that already has warm KV state for that session, reducing the
    /// cost of reloading from scratch. Keys are opaque byte strings provided
    /// by the caller (e.g. a conversation ID hash).
    kv_affinity: Mutex<HashMap<String, Rank>>,
}

impl DisaggRouter {
    /// Create a router connected to the given cluster.
    pub fn new(client: Arc<NexarClient>, config: DisaggConfig) -> Self {
        let prefill_loads = config
            .prefill_workers
            .iter()
            .map(|&rank| {
                Arc::new(PrefillLoad {
                    rank,
                    in_flight: AtomicU64::new(0),
                })
            })
            .collect();

        Self {
            client,
            config,
            next_request_id: AtomicU64::new(1),
            prefill_loads,
            decode_cursor: AtomicU64::new(0),
            kv_affinity: Mutex::new(HashMap::new()),
        }
    }

    /// Assign a prefill request to the least-loaded prefill worker.
    ///
    /// Returns the rank chosen and the request ID that was allocated.
    fn choose_prefill_worker(&self) -> (Rank, u64) {
        let load = self
            .prefill_loads
            .iter()
            .min_by_key(|pl| pl.in_flight.load(Ordering::Relaxed))
            .expect("at least one prefill worker must be configured");

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        load.in_flight.fetch_add(1, Ordering::Relaxed);

        (load.rank, request_id)
    }

    /// Decrement the in-flight counter for the given prefill worker rank.
    fn release_prefill_worker(&self, rank: Rank) {
        if let Some(pl) = self.prefill_loads.iter().find(|pl| pl.rank == rank) {
            pl.in_flight.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Choose a decode worker for a request.
    ///
    /// If `session_key` is `Some` and a warm affinity is recorded for it, that
    /// decode worker is returned.  Otherwise the router picks via round-robin.
    fn choose_decode_worker(&self, session_key: Option<&str>) -> Rank {
        if let Some(key) = session_key {
            let affinity = self.kv_affinity.lock().expect("kv_affinity mutex poisoned");
            if let Some(&rank) = affinity.get(key) {
                return rank;
            }
        }

        let n = self.config.decode_workers.len() as u64;
        let idx = self.decode_cursor.fetch_add(1, Ordering::Relaxed) % n;
        self.config.decode_workers[idx as usize]
    }

    /// Record KV cache affinity so future requests in the same session are
    /// routed to the decode worker that already has warm state.
    pub fn record_kv_affinity(&self, session_key: String, decode_rank: Rank) {
        self.kv_affinity
            .lock()
            .expect("kv_affinity mutex poisoned")
            .insert(session_key, decode_rank);
    }

    /// Evict the KV affinity record for a session (e.g. after session ends or
    /// the decode worker is restarted).
    pub fn evict_kv_affinity(&self, session_key: &str) {
        self.kv_affinity
            .lock()
            .expect("kv_affinity mutex poisoned")
            .remove(session_key);
    }

    /// Route a single prefill+decode request end-to-end.
    ///
    /// # Parameters
    /// - `token_ids_bytes`: prompt token IDs encoded as little-endian `i64` bytes.
    /// - `seq_len`: number of prompt tokens.
    /// - `max_new_tokens`: maximum tokens to generate.
    /// - `session_key`: optional session identifier for KV cache affinity routing.
    ///
    /// # Returns
    /// All generated token IDs in order.
    pub async fn route_request(
        &self,
        token_ids_bytes: &[u8],
        seq_len: u32,
        max_new_tokens: u32,
        session_key: Option<&str>,
    ) -> Result<Vec<i64>> {
        if self.config.prefill_workers.is_empty() {
            return Err(anyhow!("No prefill workers configured"));
        }
        if self.config.decode_workers.is_empty() {
            return Err(anyhow!("No decode workers configured"));
        }

        let decode_rank = self.choose_decode_worker(session_key);
        let (prefill_rank, request_id) = self.choose_prefill_worker();

        tracing::debug!(
            request_id,
            prefill_rank,
            decode_rank,
            seq_len,
            "Routing prefill request"
        );

        // Send token IDs to the prefill worker so it can start.
        transport::send_bytes(
            &self.client,
            token_ids_bytes,
            prefill_rank,
            transport::tags::ACTIVATION,
        )
        .await?;

        // Send the prefill request descriptor.
        let prefill_req = PrefillRequest {
            request_id,
            seq_len,
            decode_rank,
        };
        transport::send_bytes(
            &self.client,
            &prefill_req.to_bytes(),
            prefill_rank,
            tags::PREFILL_REQUEST,
        )
        .await?;

        // Wait for the prefill worker to confirm it has finished and transferred
        // the KV cache.
        let mut done_buf = [0u8; 16];
        transport::recv_bytes(
            &self.client,
            &mut done_buf,
            prefill_rank,
            tags::PREFILL_DONE,
        )
        .await?;
        let prefill_done = PrefillDone::from_bytes(&done_buf);
        self.release_prefill_worker(prefill_rank);

        tracing::debug!(
            request_id = prefill_done.request_id,
            kv_bytes = prefill_done.kv_bytes,
            "Prefill complete; starting decode"
        );

        // Tell the decode worker to start generating.
        let decode_req = DecodeRequest {
            request_id,
            max_new_tokens,
        };
        transport::send_bytes(
            &self.client,
            &decode_req.to_bytes(),
            decode_rank,
            tags::DECODE_REQUEST,
        )
        .await?;

        // Collect generated tokens until the decode worker signals completion.
        let mut tokens = Vec::new();
        loop {
            let mut token_buf = [0u8; 16];
            match transport::recv_bytes(
                &self.client,
                &mut token_buf,
                decode_rank,
                tags::DECODE_TOKEN,
            )
            .await
            {
                Ok(()) => {
                    let decoded = DecodedToken::from_bytes(&token_buf);
                    tokens.push(decoded.token_id);
                }
                Err(_) => {
                    let mut done_buf2 = [0u8; 16];
                    let _ = transport::recv_bytes(
                        &self.client,
                        &mut done_buf2,
                        decode_rank,
                        tags::DECODE_DONE,
                    )
                    .await;
                    break;
                }
            }
        }

        // Update KV affinity for the session.
        if let Some(key) = session_key {
            self.record_kv_affinity(key.to_string(), decode_rank);
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_choose_prefill_least_loaded() {
        let loads: Vec<Arc<PrefillLoad>> = vec![
            Arc::new(PrefillLoad {
                rank: 1,
                in_flight: AtomicU64::new(5),
            }),
            Arc::new(PrefillLoad {
                rank: 2,
                in_flight: AtomicU64::new(1),
            }),
            Arc::new(PrefillLoad {
                rank: 3,
                in_flight: AtomicU64::new(3),
            }),
        ];

        let least = loads
            .iter()
            .min_by_key(|pl| pl.in_flight.load(Ordering::Relaxed))
            .unwrap();

        assert_eq!(least.rank, 2);
    }

    #[test]
    fn test_router_kv_affinity() {
        let affinity: Mutex<HashMap<String, Rank>> = Mutex::new(HashMap::new());

        affinity
            .lock()
            .unwrap()
            .insert("session-abc".to_string(), 4);

        let rank = *affinity.lock().unwrap().get("session-abc").unwrap();
        assert_eq!(rank, 4);

        affinity.lock().unwrap().remove("session-abc");
        assert!(affinity.lock().unwrap().get("session-abc").is_none());
    }

    #[test]
    fn test_decode_cursor_round_robin() {
        let cursor = AtomicU64::new(0);
        let workers = [10u32, 20u32, 30u32];
        let n = workers.len() as u64;

        let picks: Vec<u32> = (0..6)
            .map(|_| workers[(cursor.fetch_add(1, Ordering::Relaxed) % n) as usize])
            .collect();

        assert_eq!(picks, vec![10, 20, 30, 10, 20, 30]);
    }
}
