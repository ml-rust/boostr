//! Shared test fixture for the paged KV cache wire-format tests.

use super::codec::{PAGED_MAGIC, WIRE_VERSION};
use crate::DType;

/// Build the 12-byte `[magic][version][dtype_tag]` prefix a paged buffer starts with.
pub(super) fn paged_header(dtype: DType) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(dtype as u32).to_le_bytes());
    bytes
}
