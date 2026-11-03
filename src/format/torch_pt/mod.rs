//! Narrow reader for PyTorch `.pt` / `.pth` files holding tensors: a bare
//! tensor, or a dict (nested dicts allowed) of tensors and `bool`, `int`,
//! `float` and `str` leaves, such as a state dict saved beside its
//! settings.
//!
//! The file (PyTorch ≥ 1.6) is a ZIP container: `{archive}/data.pkl` holds
//! the pickled object graph, `{archive}/data/{i}` the raw bytes of storage
//! `i`. A minimal pickle VM covers the opcodes `torch.save` emits and
//! intercepts `torch._utils._rebuild_tensor_v2` and the storage globals to
//! read dtype, shape, strides, offset and storage without running Python.
//! Anything else is a parse error: no user-supplied code ever runs.
//!
//! [`TorchPthSource`] presents a state dict as a
//! [`WeightSource`](crate::format::weight_source::WeightSource) under one
//! root prefix, for any model.

mod api;
mod io;
mod pickle;
mod reader;
mod reduce;
mod source;
mod types;
mod view;

pub use api::{TorchStateDict, load_tensor_pt, load_voice_pt};
pub use source::{ROOT_PREFIXES, TorchPthSource, detect_root};
pub use types::PtScalar;
