//! Gated full-attention block for `qwen35`: [`Qwen35AttentionBlock`]
//! (`layer`), its KV-cached inference forward (`forward`), and the CUDA
//! graph-mode decode step (`graph_mode`).

mod forward;
mod graph_mode;
mod layer;
mod projections;

pub use layer::{Qwen35AttentionBlock, Qwen35AttentionWeights};
