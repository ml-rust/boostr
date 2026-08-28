//! TCF (Tensor Contract Format) reader.
//!
//! See `hats/tcf/SPECIFICATION.md` for the normative format definition.

pub mod decode;
pub mod error;
pub mod loader;
pub mod metadata;

#[cfg(test)]
mod fixtures;

pub use decode::{decode_tensor_f32, element_count};
pub use error::{tcf_error, tcf_tensor_error};
pub use loader::TcfLoader;
pub use metadata::{TcfHeaderInfo, TcfModuleInfo, TcfTensorInfo, encoding_name};
