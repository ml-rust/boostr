//! Paged KV cache serialization for disaggregated prefill/decode transfer.
//!
//! `codec` documents the `LayeredPagedKvCache` wire format and its magic.

mod codec;
mod deserialize;
mod serialize;

#[cfg(test)]
mod test_support;

pub(in crate::distributed::inference) use codec::{
    FLAT_MAGIC, HEADER_LEN, append_le_elements, read_header, tensor_from_le_wire,
    unsupported_dtype_err, write_header,
};
#[cfg(test)]
pub(in crate::distributed::inference) use codec::{PAGED_MAGIC, WIRE_VERSION};
pub use deserialize::{PagedLayerData, deserialize_paged_kv_cache};
pub use serialize::serialize_paged_kv_cache;
