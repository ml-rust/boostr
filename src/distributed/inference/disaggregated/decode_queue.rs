//! The decode worker's bounded handoff queue.
//!
//! A handoff takes a queue place when it arrives, before decode acks it.
//! Its place frees when the decode loop takes it out. A handoff with a KV
//! cache above `max_kv_transfer_bytes`, or one that finds the queue full,
//! takes no place: decode sends the router an error event for it.

use anyhow::{Result, bail};
use std::sync::{Arc, Mutex, MutexGuard};

use super::config::DecodeQueueLimits;
use super::protocol::KvHandoff;

/// What waits in the queue now.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Usage {
    requests: usize,
    bytes: u64,
}

/// The admission state of one decode worker's queue.
#[derive(Debug)]
pub(crate) struct DecodeQueue {
    limits: DecodeQueueLimits,
    max_kv_transfer_bytes: usize,
    usage: Mutex<Usage>,
}

/// A handoff with a queue place. Dropping it, or [`Self::take`], frees the place.
pub(crate) struct Queued {
    pub(crate) handoff: KvHandoff,
    _place: Place,
}

impl Queued {
    /// Takes the handoff out of the queue and frees its place.
    pub(crate) fn take(self) -> KvHandoff {
        self.handoff
    }
}

/// One queue place. Dropping it frees the place.
struct Place {
    queue: Arc<DecodeQueue>,
    bytes: u64,
}

impl Drop for Place {
    fn drop(&mut self) {
        let mut usage = self.queue.lock();
        usage.requests = usage.requests.saturating_sub(1);
        usage.bytes = usage.bytes.saturating_sub(self.bytes);
    }
}

impl DecodeQueue {
    pub(crate) fn new(limits: DecodeQueueLimits, max_kv_transfer_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            limits,
            max_kv_transfer_bytes,
            usage: Mutex::new(Usage::default()),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Usage> {
        // Every update writes both fields at once, so a panic elsewhere leaves them consistent.
        self.usage.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Gives `handoff` a queue place.
    ///
    /// # Errors
    ///
    /// The KV cache is above `max_kv_transfer_bytes`, or the queue holds
    /// `max_requests` handoffs, or the cache does not fit in `max_bytes`
    /// beside the handoffs that wait. An empty queue takes any cache within
    /// `max_kv_transfer_bytes`.
    pub(crate) fn admit(self: &Arc<Self>, handoff: KvHandoff) -> Result<Queued> {
        let len = handoff.kv_cache.len();
        if len > self.max_kv_transfer_bytes {
            bail!(
                "the KV cache is {len} bytes, above this decode worker's {}-byte limit: raise \
                 max_kv_transfer_bytes or shorten the prompt",
                self.max_kv_transfer_bytes
            );
        }
        let bytes = len as u64;
        let mut usage = self.lock();
        if usage.requests >= self.limits.max_requests {
            bail!(
                "the decode queue is full: {} handoff(s) wait, the limit is {}. Send the \
                 request again later, or raise decode_queue.max_requests",
                usage.requests,
                self.limits.max_requests
            );
        }
        if usage.requests > 0 && usage.bytes.saturating_add(bytes) > self.limits.max_bytes {
            bail!(
                "the decode queue is full: {} KV cache bytes wait, and {bytes} more pass the \
                 {}-byte limit. Send the request again later, or raise decode_queue.max_bytes",
                usage.bytes,
                self.limits.max_bytes
            );
        }
        usage.requests += 1;
        usage.bytes = usage.bytes.saturating_add(bytes);
        drop(usage);
        Ok(Queued {
            handoff,
            _place: Place {
                queue: Arc::clone(self),
                bytes,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::GenerationParams;
    use super::*;

    fn handoff(kv_bytes: usize) -> KvHandoff {
        KvHandoff {
            request_id: 1,
            params: GenerationParams::greedy(4),
            generated: 1,
            context: vec![1, 2],
            kv_cache: vec![0; kv_bytes],
        }
    }

    fn usage(q: &DecodeQueue) -> (usize, u64) {
        let u = *q.lock();
        (u.requests, u.bytes)
    }

    #[test]
    fn the_count_bound_refuses_and_a_taken_place_frees() {
        let q = DecodeQueue::new(
            DecodeQueueLimits {
                max_requests: 2,
                max_bytes: 1 << 20,
            },
            1 << 20,
        );
        let a = q.admit(handoff(4)).unwrap();
        let b = q.admit(handoff(4)).unwrap();
        assert_eq!(usage(&q), (2, 8));
        let err = q.admit(handoff(4)).err().unwrap().to_string();
        assert!(
            err.contains("decode queue is full") && err.contains("max_requests"),
            "{err}"
        );
        drop(a);
        assert_eq!(usage(&q), (1, 4));
        let _c = q.admit(handoff(4)).unwrap();
        drop(b);
        assert_eq!(usage(&q), (1, 4));
    }

    #[test]
    fn the_byte_bound_refuses_but_an_empty_queue_takes_a_large_cache() {
        let q = DecodeQueue::new(
            DecodeQueueLimits {
                max_requests: 8,
                max_bytes: 10,
            },
            100,
        );
        let big = q.admit(handoff(100)).unwrap();
        assert_eq!(usage(&q), (1, 100));
        let err = q.admit(handoff(1)).err().unwrap().to_string();
        assert!(err.contains("max_bytes"), "{err}");
        drop(big);
        let a = q.admit(handoff(8)).unwrap();
        assert!(
            q.admit(handoff(4)).is_err(),
            "8 + 4 bytes pass the 10-byte bound"
        );
        let _b = q.admit(handoff(2)).unwrap();
        assert_eq!(usage(&q), (2, 10));
        drop(a);
        assert_eq!(usage(&q), (1, 2));
    }

    #[test]
    fn a_cache_above_the_transfer_limit_is_refused_even_when_the_queue_is_empty() {
        let q = DecodeQueue::new(DecodeQueueLimits::default(), 16);
        let err = q.admit(handoff(17)).err().unwrap().to_string();
        assert!(
            err.contains("17 bytes") && err.contains("16-byte limit"),
            "{err}"
        );
        assert_eq!(usage(&q), (0, 0));
    }
}
