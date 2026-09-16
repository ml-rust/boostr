//! KV cache serialization for disaggregated prefill/decode transfer.
//!
//! Converts a `LayeredKvCache` into a flat byte buffer suitable for sending
//! over nexar, and reconstructs it on the other side.
//!
//! # Wire format — `LayeredKvCache`
//!
//! ```text
//! [magic:       u32 LE = 0xB0057B01]   — flat (non-paged) KV cache
//! [version:     u32 LE = 1]
//! [dtype_tag:   u32 LE]                — numr DType discriminant: 1 = f32, 2 = f16, 3 = bf16
//! [num_layers:  u32 LE]
//! [seq_len:     u32 LE]       — used token count (same for all layers)
//! For each layer:
//!   [batch_size:    u32 LE]
//!   [num_kv_heads:  u32 LE]
//!   [head_dim:      u32 LE]
//!   [k_data:        seq_len * batch_size * num_kv_heads * head_dim * E bytes (LE)]
//!   [v_data:        seq_len * batch_size * num_kv_heads * head_dim * E bytes (LE)]
//! ```
//!
//! `E` is the element width the dtype tag names: 4 bytes for f32, 2 for f16 and bf16.
//! Elements are little-endian on every host and are decoded element-wise, so a received
//! buffer needs no alignment.
//!
//! The dtype tag round-trips: a bf16 cache is written as bf16 elements and comes back as
//! a bf16 cache. Only dtypes carried end to end are accepted — f32, f16 and bf16. Any
//! other dtype is refused by [`serialize_kv_cache`], naming itself, rather than
//! reinterpreted or converted, since either hands the peer numbers it cannot know are
//! wrong.
//!
//! The magic differs from the paged cache's `0xB0057B02`, so a paged buffer handed to
//! [`deserialize_kv_cache`] errors on the first four bytes.
//!
//! Both magics sit above `0xB0057B00` = 2_952_003_840. The pre-header format started
//! straight at `num_layers`, and no cache has billions of layers — every layer allocates
//! its own K and V tensors — so an old buffer can never present a matching magic. It is
//! rejected up front instead of being misparsed.

mod deserialize;
mod serialize;

pub use deserialize::deserialize_kv_cache;
pub use serialize::serialize_kv_cache;
