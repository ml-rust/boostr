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

/// A non-tensor leaf of a saved dict: a Python `bool`, `int`, `float` or
/// `str`.
#[derive(Debug, Clone, PartialEq)]
pub enum PtScalar {
    /// A Python `bool`.
    Bool(bool),
    /// A Python `int` that fits in 64 bits.
    Int(i64),
    /// A Python `float`.
    Float(f64),
    /// A Python `str`.
    Str(String),
}

impl PtScalar {
    /// The value as a number: a `float`, or an `int` widened to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
}

/// Result of parsing a `.pt` file.
#[derive(Debug, Clone)]
pub(super) struct PtContents {
    /// Either the bare tensor ("" key) or a flat map of `name -> meta` if the
    /// top-level was a dict.
    pub tensors: HashMap<String, PtTensorMeta>,
    /// The scalar leaves of a top-level dict, by the same dotted names.
    pub scalars: HashMap<String, PtScalar>,
}
