//! LoRA (Low-Rank Adaptation) layer.
//!
//! Adds a low-rank A*B decomposition to an existing linear layer:
//! output = base_linear(x) + (x @ A^T) @ B^T * scaling
//!
//! where A: [rank, in_features], B: [out_features, rank], and scaling is
//! alpha / rank or a per-module value.
//!
//! - `layer`: the [`LoraLinear`] type, constructors, accessors, in-place
//!   adapter updates
//! - `scaled`: construction from loaded factors with a given scaling
//! - `fused`: one block-diagonal adapter over a row-concatenated base
//! - `cast`: loaded factors cast to the dtype the adapter computes in
//! - `forward`: the tracked forward pass
//! - `merge`: folding the adapter into a dense base
//! - `module`: the [`Module`](crate::nn::module::Module) impl

mod cast;
mod forward;
mod fused;
mod layer;
mod merge;
mod module;
mod scaled;

pub use fused::{FusedLoraPart, LoraFactors};
pub use layer::LoraLinear;
