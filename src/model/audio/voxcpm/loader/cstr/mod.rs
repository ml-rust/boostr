//! Reading a VoxCPM2 GGUF whose tensors carry llama.cpp's ggml-conventional
//! names (as in `cstr/voxcpm2-GGUF`) instead of the checkpoint's own
//! HuggingFace names. Transformer stack only — the AudioVAE is out of scope,
//! see [`GgmlNamedGguf`].

mod names;
mod source;
#[cfg(test)]
mod tests;

pub use names::{GGML_SENTINEL, hf_to_ggml_name};
pub(crate) use names::{GgufNaming, probe_naming};
pub(crate) use source::GgmlNamedGguf;
