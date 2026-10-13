//! Causal GQA attention for VoxCPM2's MiniCPM4 decoder.
//!
//! `block` documents the full-sequence causal path, the optional RoPE tables
//! and `head_dim`. `cached` documents the KV-cached path.
//!
//! - `block`: [`MiniCpm4Attention`], the full-sequence forward, `alias`, and
//!   the `Module` impl
//! - `cached`: the KV-cached `forward_cached`
//! - `guards`: preconditions the cached path turns into loud errors
//! - `lora`: adapter attachment and write-back

mod block;
mod cached;
mod guards;
mod lora;

pub use block::MiniCpm4Attention;
