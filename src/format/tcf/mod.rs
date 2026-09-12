//! TCF (Tensor Contract Format) model loader.
//!
//! The container itself lives in [`crate::tcf`]; `src/tcf/FORMAT.md` is the
//! layout reference. This module turns its records into `QuantTensor`s.

pub mod block;
pub mod decode;
pub mod error;
pub mod loader;
pub mod metadata;

#[cfg(test)]
pub(crate) mod fixtures;

pub use block::{BoostrBlockDecoder, block_format, decode_block_f32};
pub use decode::{decode_tensor_f32, element_count};
pub use error::{tcf_error, tcf_tensor_error};
pub use loader::{TcfLoader, TcfSession};
pub use metadata::{TcfHeaderInfo, TcfModuleInfo, TcfTensorInfo, encoding_name};
