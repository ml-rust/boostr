//! Wire protocol for disaggregated prefill/decode inference.
//!
//! Separates compute-heavy prefill from memory-bound decode across workers.
//! After prefill completes, the KV cache state is transferred to the target
//! decode worker via nexar transport, and that worker continues autoregressive
//! generation independently.
//!
//! # Architecture
//!
//! ```text
//!          ┌─────────────┐
//!  HTTP    │             │  round-robin (least loaded)
//!  ───────►│   Router    │──────────────────────────────►  Prefill Worker
//!          │             │
//!          └──────┬──────┘
//!                 │ cache-aware routing (warm KV affinity)
//!                 ▼
//!          Decode Worker  ◄──────  KV cache transfer (nexar)
//! ```
//!
//! # Message tags (disaggregated protocol)
//!
//! | Tag | Meaning |
//! |-----|---------|
//! | 30  | `PREFILL_REQUEST`  – router → prefill worker |
//! | 31  | `PREFILL_DONE`     – prefill worker → router |
//! | 32  | `KV_CACHE`         – prefill worker → decode worker |
//! | 33  | `KV_CACHE_ACK`     – decode worker → prefill worker (ready) |
//! | 34  | `DECODE_REQUEST`   – router → decode worker |
//! | 35  | `DECODE_TOKEN`     – decode worker → router (one token) |
//! | 36  | `DECODE_DONE`      – decode worker → router (EOS or max tokens) |

use nexar::Rank;

/// Prefill function signature: `(token_ids_bytes, seq_len) -> (activation_bytes, kv_cache_bytes)`
pub type PrefillFn = Box<dyn Fn(&[u8], usize) -> (Vec<u8>, Vec<u8>) + Send + Sync>;

/// Decode step function signature: `(kv_cache_bytes, last_token_id, position) -> (next_token_id, updated_kv_bytes)`
pub type DecodeStepFn = Box<dyn Fn(&[u8], i64, u32) -> (i64, Vec<u8>) + Send + Sync>;

/// Message tags for the disaggregated prefill/decode protocol.
pub mod tags {
    /// Router → prefill worker: start a prefill run.
    pub const PREFILL_REQUEST: u32 = 30;
    /// Prefill worker → router: prefill complete, decode worker chosen.
    pub const PREFILL_DONE: u32 = 31;
    /// Prefill worker → decode worker: serialised KV cache payload.
    pub const KV_CACHE: u32 = 32;
    /// Decode worker → prefill worker: KV cache received, ready to decode.
    pub const KV_CACHE_ACK: u32 = 33;
    /// Router → decode worker: begin decode loop.
    pub const DECODE_REQUEST: u32 = 34;
    /// Decode worker → router: one generated token (i64 LE).
    pub const DECODE_TOKEN: u32 = 35;
    /// Decode worker → router: generation finished (EOS or max tokens hit).
    pub const DECODE_DONE: u32 = 36;
}

/// Role of a disaggregated inference worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisaggRole {
    Prefill,
    Decode,
    Router,
}

/// Configuration for disaggregated inference.
#[derive(Debug, Clone)]
pub struct DisaggConfig {
    /// Ranks of all prefill workers.
    pub prefill_workers: Vec<Rank>,
    /// Ranks of all decode workers.
    pub decode_workers: Vec<Rank>,
    /// Rank of the router node.
    pub router_rank: Rank,
    /// Maximum KV cache bytes that may be transferred per request.
    /// Requests whose KV cache exceeds this limit are rejected with an error.
    pub max_kv_transfer_bytes: usize,
}

/// Wire format for a prefill request (router → prefill worker).
///
/// Serialised as: `[request_id: u64, seq_len: u32, decode_rank: u32]` = 16 bytes.
#[derive(Debug, Clone, Copy)]
pub struct PrefillRequest {
    /// Unique request identifier assigned by the router.
    pub request_id: u64,
    /// Number of prompt tokens.
    pub seq_len: u32,
    /// The decode worker rank the prefill worker should push the KV cache to.
    pub decode_rank: u32,
}

impl PrefillRequest {
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.request_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.decode_rank.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 16]) -> Self {
        Self {
            request_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            seq_len: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            decode_rank: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        }
    }
}

/// Wire format for the prefill-done acknowledgment (prefill worker → router).
///
/// Serialised as: `[request_id: u64, kv_bytes: u64]` = 16 bytes.
#[derive(Debug, Clone, Copy)]
pub struct PrefillDone {
    /// Mirrors the request ID from the prefill request.
    pub request_id: u64,
    /// Number of KV cache bytes that were transferred to the decode worker.
    pub kv_bytes: u64,
}

impl PrefillDone {
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.request_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.kv_bytes.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 16]) -> Self {
        Self {
            request_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            kv_bytes: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        }
    }
}

/// Wire format for a decode request (router → decode worker).
///
/// Serialised as: `[request_id: u64, max_new_tokens: u32, _pad: u32]` = 16 bytes.
#[derive(Debug, Clone, Copy)]
pub struct DecodeRequest {
    /// Request ID (must match the KV cache already received for this request).
    pub request_id: u64,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: u32,
}

impl DecodeRequest {
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.request_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.max_new_tokens.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 16]) -> Self {
        Self {
            request_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            max_new_tokens: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
        }
    }
}

/// Wire format for a single generated token (decode worker → router).
///
/// Serialised as: `[request_id: u64, token_id: i64]` = 16 bytes.
#[derive(Debug, Clone, Copy)]
pub struct DecodedToken {
    pub request_id: u64,
    pub token_id: i64,
}

impl DecodedToken {
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.request_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.token_id.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 16]) -> Self {
        Self {
            request_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            token_id: i64::from_le_bytes(buf[8..16].try_into().unwrap()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_request_roundtrip() {
        let orig = PrefillRequest {
            request_id: 42,
            seq_len: 128,
            decode_rank: 3,
        };
        let bytes = orig.to_bytes();
        let decoded = PrefillRequest::from_bytes(&bytes);
        assert_eq!(decoded.request_id, 42);
        assert_eq!(decoded.seq_len, 128);
        assert_eq!(decoded.decode_rank, 3);
    }

    #[test]
    fn test_prefill_done_roundtrip() {
        let orig = PrefillDone {
            request_id: 99,
            kv_bytes: 1_048_576,
        };
        let bytes = orig.to_bytes();
        let decoded = PrefillDone::from_bytes(&bytes);
        assert_eq!(decoded.request_id, 99);
        assert_eq!(decoded.kv_bytes, 1_048_576);
    }

    #[test]
    fn test_decode_request_roundtrip() {
        let orig = DecodeRequest {
            request_id: 7,
            max_new_tokens: 512,
        };
        let bytes = orig.to_bytes();
        let decoded = DecodeRequest::from_bytes(&bytes);
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.max_new_tokens, 512);
    }

    #[test]
    fn test_decoded_token_roundtrip() {
        let orig = DecodedToken {
            request_id: 1,
            token_id: 12345,
        };
        let bytes = orig.to_bytes();
        let decoded = DecodedToken::from_bytes(&bytes);
        assert_eq!(decoded.request_id, 1);
        assert_eq!(decoded.token_id, 12345);
    }
}
