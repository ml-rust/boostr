//! Shared utilities for Flash Attention v2: parameter validation, block config,
//! shared memory helpers.
//!
//! Wiring only. The implementations live in three siblings, split by concern:
//! - `flash_params.rs`: `AttentionParams` and `validate_qkv`
//! - `flash_smem.rs`: device shared-memory query, sizing formulas, opt-in attribute
//! - `flash_block_config.rs`: forward/backward tile tables and selection
//!
//! Everything callers imported from `flash_utils` is re-exported here, so
//! `super::flash_utils::{...}` paths across the attention module are unchanged.

pub(super) use super::flash_block_config::bwd_block_config;
pub(super) use super::flash_params::{AttentionParams, validate_qkv};
pub(crate) use super::flash_smem::set_smem_attribute;
// Widened past `pub(super)`: these are consumed by paged/varlen/mla/mqa_gqa,
// which are siblings of `flash`, not of this file.
pub(in crate::ops::cuda::attention) use super::flash_smem::{
    compute_bwd_smem, compute_smem, device_max_smem,
};
