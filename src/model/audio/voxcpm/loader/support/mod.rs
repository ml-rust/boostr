//! Shared tensor-fetch and sub-module-assembly helpers for the VoxCPM2
//! encoder/decoder loaders.
//!
//! Same idiom as `neucodec/loader/support.rs` (not reused directly: that
//! module's helper is private to its own `loader` submodule).

mod tcf;
mod tensor_loader;
#[cfg(test)]
mod tests;
mod torch_pth;
mod weight_source;

pub use tcf::TcfSource;
pub(crate) use tensor_loader::TensorLoader;
pub use torch_pth::TorchPthSource;
pub use weight_source::WeightSource;
