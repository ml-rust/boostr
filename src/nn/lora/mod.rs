//! LoRA (Low-Rank Adaptation) layer.
//!
//! Adds a low-rank A*B decomposition to an existing linear layer:
//! output = base_linear(x) + (x @ A^T) @ B^T * scaling
//!
//! where A: [rank, in_features], B: [out_features, rank], scaling = alpha / rank.
//!
//! - `layer`: the [`LoraLinear`] type, constructors, accessors, in-place
//!   adapter updates
//! - `forward`: the tracked forward pass
//! - `merge`: folding the adapter into a dense base
//! - `module`: the [`Module`](crate::nn::module::Module) impl

mod forward;
mod layer;
mod merge;
mod module;

pub use layer::LoraLinear;
