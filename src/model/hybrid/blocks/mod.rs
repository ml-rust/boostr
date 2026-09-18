//! AttentionBlock, SsmBlock and GdnBlock sub-modules for the hybrid model.
//!
//! - `attention`: the [`AttentionBlock`] type and its KV-cached forward
//! - `mask`: [`AttentionBlock::attention_mask`], the causal / windowed / ALiBi mask
//! - `ssm`: the [`SsmBlock`] type and its inference forward
//! - `gdn`: the [`GdnBlock`] Gated DeltaNet mixer (`qwen35`) and its forward
//! - `attention_qwen35`: the [`Qwen35AttentionBlock`] gated full-attention
//!   mixer (`qwen35`) and its KV-cached forward

mod attention;
mod attention_qwen35;
mod gdn;
mod mask;
mod ssm;

pub(super) use attention::AttentionBlock;
pub use attention_qwen35::{Qwen35AttentionBlock, Qwen35AttentionWeights};
pub use gdn::{GdnBlock, GdnWeights};
pub(super) use ssm::SsmBlock;
