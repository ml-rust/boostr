//! Hybrid model mixing Llama attention and Mamba2 SSM blocks.
//!
//! - `build`: the [`HybridModel`] type and [`HybridModel::from_varbuilder`]
//! - `forward`: the KV-cached forward, the hidden-state forward, and accessors

mod build;
mod forward;

pub use build::HybridModel;
