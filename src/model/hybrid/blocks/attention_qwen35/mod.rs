//! Gated full-attention block for `qwen35`: [`Qwen35AttentionBlock`]
//! (`layer`) and its KV-cached inference forward (`forward`).

mod forward;
mod layer;
mod projections;

pub use layer::{Qwen35AttentionBlock, Qwen35AttentionWeights};
