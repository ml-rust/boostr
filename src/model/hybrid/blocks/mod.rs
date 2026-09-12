//! AttentionBlock and SsmBlock sub-modules for the hybrid model.
//!
//! - `attention`: the [`AttentionBlock`] type and its KV-cached forward
//! - `mask`: [`AttentionBlock::attention_mask`], the causal / windowed / ALiBi mask
//! - `ssm`: the [`SsmBlock`] type and its inference forward

mod attention;
mod mask;
mod ssm;

pub(super) use attention::AttentionBlock;
pub(super) use ssm::SsmBlock;
