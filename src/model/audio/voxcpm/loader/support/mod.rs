//! Shared tensor-fetch and sub-module-assembly helpers for the VoxCPM2
//! encoder/decoder loaders.
//!
//! Same idiom as `neucodec/loader/support.rs` (not reused directly: that
//! module's helper is private to its own `loader` submodule).
//!
//! [`WeightSource`], [`DenseWeightSource`] and [`TcfSource`] now live in
//! [`crate::format::weight_source`] — they are format concerns, and a
//! text-model loader outside the `audio` feature needs them too. They are
//! re-exported here so every VoxCPM2 sub-loader keeps its existing import
//! path.

mod tensor_loader;
#[cfg(test)]
mod tests;
mod torch_pth;

pub use crate::format::weight_source::{DenseWeightSource, TcfSource, WeightSource};
pub(crate) use tensor_loader::TensorLoader;
pub use torch_pth::TorchPthSource;
