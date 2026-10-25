//! nexar message tags of the disaggregated protocol.
//!
//! Frames travel over `NexarClient::send_bytes_tagged`. nexar keeps one
//! receive channel per `(source rank, tag)`, so frames of one tag never
//! consume frames of another. nexar numbers its own collective tags upward
//! from 1. These tags sit in a fixed high range that never meets them.

/// Shared prefix of every disaggregated tag: ASCII `DISAGG` in the top six bytes.
const BASE: u64 = 0x4449_5341_4747_0000;

/// Router → prefill worker: a `PrefillRequest` frame.
pub const PREFILL_REQUEST: u64 = BASE | 1;
/// Prefill worker → decode worker: a `KvHandoff` frame.
pub const KV_HANDOFF: u64 = BASE | 2;
/// Router → prefill or decode worker: a `Cancel` frame.
pub const CANCEL: u64 = BASE | 3;
/// Prefill or decode worker → router: an `Event` frame.
pub const EVENT: u64 = BASE | 4;
/// Router → prefill or decode worker: stop the run loop. A one-byte frame
/// outside the versioned frame format: the tag alone carries the message.
pub const SHUTDOWN: u64 = BASE | 5;
