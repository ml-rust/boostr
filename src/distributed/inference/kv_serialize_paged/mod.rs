//! Paged KV cache serialization for disaggregated prefill/decode transfer.
//!
//! # Wire format — `LayeredPagedKvCache`
//!
//! ```text
//! [magic:       u32 LE = 0xB0057B02]   — paged KV cache
//! [version:     u32 LE = 1]
//! [dtype_tag:   u32 LE]                — numr DType discriminant: 1 = f32, 2 = f16, 3 = bf16
//! [num_layers:  u32 LE]
//! [block_size:  u32 LE]
//! [seq_len:     u32 LE]
//! For each layer:
//!   [num_blocks:    u32 LE]
//!   [num_heads:     u32 LE]
//!   [head_dim:      u32 LE]
//!   [k_data:        num_blocks * block_size * num_heads * head_dim * E bytes (LE)]
//!   [v_data:        num_blocks * block_size * num_heads * head_dim * E bytes (LE)]
//!   [block_table_len: u32 LE]
//!   [block_ids:     block_table_len * 4 bytes (u32 LE)]   — BlockId = u32
//! ```
//!
//! `E` is the element width the dtype tag names: 4 bytes for f32, 2 for f16 and bf16.
//! Elements are little-endian on every host and are decoded element-wise, so a received
//! buffer needs no alignment.
//!
//! The magic differs from the flat cache's `0xB0057B01`, so a paged buffer handed to
//! [`super::kv_serialize::deserialize_kv_cache`] errors on the first four bytes instead
//! of reading block-table fields as layer dimensions.
//!
//! Both magics sit above `0xB0057B00` = 2_952_003_840. The pre-header format started
//! straight at `num_layers`, and no cache has billions of layers — every layer
//! allocates its own K and V tensors — so an old buffer can never present a matching
//! magic. It is rejected up front instead of being misparsed.

mod codec;
mod deserialize;
mod serialize;

#[cfg(test)]
mod test_support;

pub(in crate::distributed::inference) use codec::{
    FLAT_MAGIC, HEADER_LEN, PAGED_MAGIC, WIRE_VERSION, append_le_elements, dtype_from_tag,
    dtype_to_tag, le_wire_to_native_bytes, read_f32_le_vec, read_header, read_le_f32_widened,
    tensor_from_le_wire, unsupported_dtype_err, write_header,
};
pub use deserialize::{PagedLayerData, deserialize_paged_kv_cache};
pub use serialize::serialize_paged_kv_cache;
