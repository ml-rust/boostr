//! Parsed metadata types shared between the pickle VM and the loaders.

use numr::dtype::DType;
use std::collections::HashMap;

/// Parsed tensor metadata extracted from `data.pkl`.
#[derive(Debug, Clone)]
pub(super) struct PtTensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub storage_id: String,
    pub storage_offset: usize,
    /// Total number of elements in the underlying storage (used to validate the
    /// raw bytes length in `data/{storage_id}`).
    pub storage_numel: usize,
    /// Bytes per element for the storage's dtype (may differ from `dtype` if
    /// the tensor is a view; not the case for simple `torch.save(tensor)`).
    pub storage_elem_size: usize,
}

/// Result of parsing a `.pt` file.
#[derive(Debug, Clone)]
pub(super) struct PtContents {
    /// Either the bare tensor ("" key) or a flat map of `name -> meta` if the
    /// top-level was a dict.
    pub tensors: HashMap<String, PtTensorMeta>,
}
