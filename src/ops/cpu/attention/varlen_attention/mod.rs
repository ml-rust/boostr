//! CPU fallback for variable-length (ragged) attention
//!
//! Unpacks sequences from the packed buffer, runs standard attention per-sequence,
//! and writes results back to the packed output buffer.

mod backward;
mod dispatch;
mod forward;
