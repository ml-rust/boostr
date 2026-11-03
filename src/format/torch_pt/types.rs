//! Parsed metadata types shared between the pickle VM and the loaders.

use numr::dtype::DType;
use std::collections::HashMap;

/// Parsed tensor metadata extracted from `data.pkl`.
#[derive(Debug, Clone)]
pub(super) struct PtTensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Element strides as saved, one per dim of `shape`.
    pub strides: Vec<usize>,
    pub storage_id: String,
    pub storage_offset: usize,
    /// Total number of elements in the underlying storage (used to validate the
    /// raw bytes length in `data/{storage_id}`).
    pub storage_numel: usize,
    /// Bytes per element for the storage's dtype (may differ from `dtype` if
    /// the tensor is a view; not the case for simple `torch.save(tensor)`).
    pub storage_elem_size: usize,
}

/// A value on the pickle VM stack.
#[derive(Debug, Clone)]
pub(super) enum PValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<PValue>),
    List(Vec<PValue>),
    Dict(Vec<(PValue, PValue)>),
    /// `(module, qualname)` reference to a Python callable; used to intercept
    /// `torch._utils._rebuild_tensor_v2`, dtype objects, etc.
    Global(String, String),
    /// `persistent_load(pid)` — for tensors this is `("storage", dtype_global,
    /// storage_id, device, numel)` where storage_id is the filename in
    /// `data/*`.
    PersistentRef {
        dtype: DType,
        storage_id: String,
        numel: usize,
    },
    /// A `REDUCE`d tensor. Fully resolved; holds the metadata we care about.
    Tensor(PtTensorMeta),
    /// Placeholder for things we don't interpret (e.g. OrderedDict, Size,
    /// DeviceObj).
    #[allow(dead_code)]
    Opaque(String, String),
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
