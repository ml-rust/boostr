//! KV cache serialization for disaggregated prefill/decode transfer.
//!
//! Converts a `LayeredKvCache` into a flat byte buffer suitable for sending
//! over nexar, and reconstructs it on the other side.
//!
//! `serialize` documents the wire format. `deserialize` documents how the
//! magic rejects a paged buffer and the pre-header format.

mod deserialize;
mod serialize;

pub use deserialize::deserialize_kv_cache;
pub use serialize::serialize_kv_cache;
