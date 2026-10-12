//! Gated full-attention block for `qwen35`: [`Qwen35AttentionBlock`]
//! (`layer`), the projection and gate steps its forwards share
//! (`projections`), its KV-cached inference forward (`forward`), the
//! forward of lanes that share one cached prefix (`shared_prefix`), the batched
//! decode step over one lane per sequence (`batched`), the CUDA
//! graph-mode decode step (`graph_mode`), and unmerged LoRA adapters
//! (`adapter`).

mod adapter;
mod batched;
mod forward;
mod graph_mode;
mod layer;
mod projections;
mod shared_prefix;

pub use adapter::Qwen35AttentionProjection;
#[cfg(feature = "cuda")]
pub use graph_mode::Qwen35GraphStep;
pub use layer::{Qwen35AttentionBlock, Qwen35AttentionWeights};
