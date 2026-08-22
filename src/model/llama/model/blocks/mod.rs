//! LLaMA building blocks — module hub.
//!
//! Code is split across sibling files:
//!   helpers.rs   — var_contiguous, repeat_kv
//!   attention.rs — LlamaAttention + all its impls
//!   mlp.rs       — LlamaMlp + its SwiGLU impl
//!   block.rs     — LlamaBlock + its forward impls
//!   builders.rs  — build_block_from_varbuilder, build_block_from_config

pub(super) mod attention;
pub(super) mod block;
pub(super) mod builders;
pub(super) mod helpers;
pub(super) mod mlp;
pub(super) mod moe;

pub(super) use block::LlamaBlock;
pub(super) use builders::{build_block_from_config, build_block_from_varbuilder};
pub use moe::{ExpertWeights, LlamaFfn, LlamaMoeMlp};
