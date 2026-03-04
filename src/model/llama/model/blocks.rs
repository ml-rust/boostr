//! LLaMA building blocks — module hub.
//!
//! Code is split across sibling files:
//!   blocks_helpers.rs  — var_contiguous, repeat_kv
//!   blocks_attention.rs — LlamaAttention + all its impls
//!   blocks_mlp.rs      — LlamaMlp + its SwiGLU impl
//!   blocks_block.rs    — LlamaBlock + its forward impls
//!   blocks_builders.rs — build_block_from_varbuilder, build_block_from_config

#[path = "blocks_helpers.rs"]
pub(super) mod helpers;

#[path = "blocks_attention.rs"]
pub(super) mod attention;

#[path = "blocks_mlp.rs"]
pub(super) mod mlp;

#[path = "blocks_block.rs"]
pub(super) mod block;

#[path = "blocks_builders.rs"]
pub(super) mod builders;

pub(super) use block::LlamaBlock;
pub(super) use builders::{build_block_from_config, build_block_from_varbuilder};
