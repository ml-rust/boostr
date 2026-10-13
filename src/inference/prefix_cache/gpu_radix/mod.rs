//! GPU-resident radix tree for O(1) amortised prefix matching.
//!
//! # Architecture
//!
//! The radix tree maps block-aligned token-sequence hashes to cached `BlockId`
//! values.  The hash table is stored as three parallel arrays on the CPU and —
//! when the `cuda` feature is enabled — mirrored into GPU global memory so that
//! batch prefix lookups can be executed entirely on the device, avoiding a
//! CPU scheduling bottleneck. `types` documents the slot layout and `ops`
//! the probing strategy and tiering.
//!
//! # Feature gating
//!
//! The `GpuRadixTree` type and its CUDA-backed `PrefixLookup` implementation
//! are compiled only when the `cuda` feature is enabled.  A pure-CPU fallback
//! is provided unconditionally so that the `PrefixLookup` trait can be used
//! in feature-agnostic scheduling code.

mod ops;
mod types;

pub use types::{GpuRadixStats, GpuRadixTree, PrefixLookup};
