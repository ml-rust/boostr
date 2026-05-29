//! Narrow reader for PyTorch `.pt` / `.pth` files containing tensors.
//!
//! Covers the two shapes we actually encounter in TTS voice files:
//!
//! * `torch.save(tensor, "f.pt")` — bare float tensor
//! * `torch.save({"style": tensor}, "f.pt")` — dict with one tensor entry
//!
//! The `.pt` format (PyTorch ≥ 1.6) is a ZIP container with three relevant
//! entries:
//!
//! * `{archive}/data.pkl` — pickle describing the object graph (dtype, shape,
//!   stride, storage references, etc.)
//! * `{archive}/data/{i}` — raw bytes for storage `i`
//! * `{archive}/version` — numeric format version (ignored here)
//!
//! We implement a minimal pickle VM that covers the opcodes produced by
//! `torch.save` on a tensor (`PROTO`, `FRAME`, `EMPTY_DICT`, `MARK`, integer /
//! string literals, `TUPLE*`, `SETITEMS`, `STACK_GLOBAL`, `REDUCE`, `BUILD`,
//! `BINPERSID`, memoization, `STOP`). We intercept
//! `torch._utils._rebuild_tensor_v2` and equivalent rebuild callables to
//! extract `(dtype, shape, storage_offset, storage_id)` without running any
//! actual Python. Everything else is a parse error — we never execute user-
//! supplied code.

mod api;
mod io;
mod pickle;
mod reader;
mod reduce;
mod types;

pub use api::{TorchStateDict, load_tensor_pt, load_voice_pt};
