//! Gated full-attention block for `qwen35`: [`Qwen35AttentionBlock`]
//! (`layer`), the projection and gate steps its forwards share
//! (`projections`), its KV-cached inference forward (`forward`), the
//! forward of lanes that share one cached prefix (`shared_prefix`), the batched
//! decode step over one lane per sequence (`batched`), the CUDA
//! graph-mode decode step (`graph_mode`), unmerged LoRA adapters
//! (`adapter`), the tracked training forward (`train`, checkpointed in
//! `train_checkpoint`), parameter access and the trainable switch
//! (`params`), and the training constructors (`construct`).

mod adapter;
mod batched;
mod construct;
mod forward;
mod graph_mode;
mod layer;
mod params;
mod projections;
mod shared_prefix;
mod train;
mod train_checkpoint;
#[cfg(test)]
mod train_fixture;

pub use adapter::Qwen35AttentionProjection;
pub use construct::Qwen35AttentionWeightsWithIds;
#[cfg(feature = "cuda")]
pub use graph_mode::Qwen35GraphStep;
pub use layer::{Qwen35AttentionBlock, Qwen35AttentionWeights};
