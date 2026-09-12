//! RoPE (Rotary Position Embedding) module
//!
//! Wraps the RoPEOps trait as a reusable module with precomputed frequency caches.
//!
//! - `table`: the [`RoPE`] cache type, forward, cast, narrow, alias
//! - `precompute`: `RoPE::precompute_freqs` and the scaling-type dispatch
//! - `scaling`: YaRN and LongRoPE frequency scaling

mod precompute;
mod scaling;
mod table;

pub use table::RoPE;
