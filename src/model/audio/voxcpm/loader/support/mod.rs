//! Shared tensor-fetch and sub-module-assembly helpers for the VoxCPM2
//! encoder/decoder loaders.
//!
//! Same idiom as `neucodec/loader/support.rs` (not reused directly: that
//! module's helper is private to its own `loader` submodule).

mod tensor_loader;
#[cfg(test)]
mod tests;
mod weight_source;

pub(crate) use tensor_loader::TensorLoader;
pub use weight_source::WeightSource;
