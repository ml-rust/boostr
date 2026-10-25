//! AttentionBlock, MlaBlock, SsmBlock and GdnBlock sub-modules for the hybrid model.
//!
//! - `attention`: the `AttentionBlock` type and its KV-cached forward
//! - `ffn`: the `FeedForward` half of an attention layer, MLP or MoE
//! - `mask`: `AttentionBlock::attention_mask`, the causal / windowed / ALiBi mask
//! - `mla`: the `MlaBlock` type, absorbed MLA over a latent KV cache
//! - `ssm`: the `SsmBlock` type and its inference forward
//! - `gdn`: the [`GdnBlock`] Gated DeltaNet mixer (`qwen35`) and its forward
//! - `attention_qwen35`: the [`Qwen35AttentionBlock`] gated full-attention
//!   mixer (`qwen35`) and its KV-cached forward

mod attention;
mod attention_qwen35;
mod ffn;
mod gdn;
mod mask;
mod mla;
mod ssm;

pub(super) use attention::AttentionBlock;
#[cfg(feature = "cuda")]
pub use attention_qwen35::Qwen35GraphStep;
pub use attention_qwen35::{
    Qwen35AttentionBlock, Qwen35AttentionProjection, Qwen35AttentionWeights,
    Qwen35AttentionWeightsWithIds,
};
pub(super) use ffn::FeedForward;
pub use gdn::{GdnBlock, GdnProjection, GdnWeights, GdnWeightsWithIds};
pub(super) use mla::MlaBlock;
pub use ssm::SSM_PRENORM_EPS;
pub(super) use ssm::SsmBlock;
