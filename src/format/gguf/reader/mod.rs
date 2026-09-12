//! GGUF file reader
//!
//! Parses the GGUF binary format (v1-v3) and provides access to metadata
//! and tensor data. F32 tensors are loaded as `Tensor<R>`, quantized tensors
//! as `QuantTensor<R>`.

mod load;
mod open;
mod streaming;

pub use open::Gguf;
