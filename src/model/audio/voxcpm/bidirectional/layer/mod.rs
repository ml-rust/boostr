//! Single pre-norm transformer layer for VoxCPM2's shared bidirectional
//! MiniCPM4 block stack, used by both `feat_encoder` (`local_encoder`) and
//! the local DiT (`feat_decoder`).
//!
//! Same pre-norm-attention-residual, pre-norm-MLP-residual sequence as
//! `LlamaBlock` (`RmsNorm` -> attn -> add, `RmsNorm` -> MLP -> add), with the
//! two differences this checkpoint requires: attention is
//! [`BidirectionalAttention`] (bidirectional GQA, not `LlamaAttention`'s
//! always-causal path) and residuals are plain adds — `use_mup` is `false`
//! on this checkpoint, so no muP `scale_depth/sqrt(num_layers)` factor is
//! applied (unlike some MiniCPM-lineage ports that assume it is).

mod forward;
mod lora_module;
mod model;

pub use model::BidirectionalLayer;
