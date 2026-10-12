//! Gated full-attention block for `qwen35`: [`Qwen35AttentionBlock`]
//! (`layer`), its KV-cached inference forward (`forward`), the batched
//! decode step over one lane per sequence (`batched`), the CUDA
//! graph-mode decode step (`graph_mode`), and unmerged LoRA adapters
//! (`adapter`).

mod adapter;
mod batched;
mod forward;
mod graph_mode;
mod layer;
mod projections;

pub use adapter::Qwen35AttentionProjection;
#[cfg(feature = "cuda")]
pub use graph_mode::Qwen35GraphStep;
pub use layer::{Qwen35AttentionBlock, Qwen35AttentionWeights};
