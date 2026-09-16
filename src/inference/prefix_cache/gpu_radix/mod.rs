//! GPU-resident radix tree for O(1) amortised prefix matching.
//!
//! # Architecture
//!
//! The radix tree maps block-aligned token-sequence hashes to cached `BlockId`
//! values.  The hash table is stored as three parallel arrays on the CPU and —
//! when the `cuda` feature is enabled — mirrored into GPU global memory so that
//! batch prefix lookups can be executed entirely on the device, avoiding a
//! CPU scheduling bottleneck.
//!
//! CPU layout (one entry per slot):
//! ```text
//! keys:      [u64; capacity]   — FNV hash of the token subsequence
//! values:    [i32; capacity]   — BlockId (-1 = empty slot)
//! metadata:  [u64; capacity]   — packed (ref_count: u32, lru_timestamp: u32)
//! ```
//!
//! Probing strategy: open addressing with linear probing.  The capacity is
//! always a power of two so that `hash & (capacity - 1)` gives the slot index.
//!
//! # Tiering
//!
//! VRAM (hot) → RAM (warm) → NVMe (cold, future work).
//! Eviction promotes entries toward cold tiers; insertion starts in VRAM.
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
