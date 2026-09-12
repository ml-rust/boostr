//! Causal GQA attention for VoxCPM2's MiniCPM4 decoder.
//!
//! Unlike the `feat_encoder` sibling in this module — the one bidirectional
//! transformer in the VoxCPM2 stack, which hand-writes its own unmasked
//! orchestration — this block is a plain causal decoder, so its full-sequence
//! [`MiniCpm4Attention::forward`] runs the shared [`attention_core_masked`]
//! sequence that every other causal block in the crate uses (`LlamaAttention`
//! included). That helper owns reshape/permute, contiguity, RoPE, the GQA head
//! repeat, and the causal mask; nothing here re-derives any of it.
//!
//! Causality is not a flag there: `attention_core_masked` always builds a
//! causal mask. That is the full-sequence forward, so without it every
//! position would attend to FUTURE positions while every shape stayed valid.
//!
//! The KV-cached [`MiniCpm4Attention::forward_cached`] instead calls the flash
//! kernel directly, as `LlamaAttention::forward_with_kv_cache` does.
//!
//! Both entry points take the RoPE tables as `Option`: VoxCPM2's
//! `residual_lm` is this same block with `no_rope` set, and its loader builds
//! no table at all. `no_rope` is NoPE — the rotation is dropped and NOTHING
//! replaces it (no ALiBi, no learned positions), so position reaches the block
//! only through causal ordering. A `None` table with `no_rope` unset is an
//! error, never a silent skip.
//!
//! `head_dim` (128) is read from config, never derived from
//! `hidden_size / num_heads` — see
//! [`MiniCpm4Config::head_dim`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config::head_dim).
//!
//! - `block`: [`MiniCpm4Attention`], the full-sequence forward, `alias`, and
//!   the `Module` impl
//! - `cached`: the KV-cached `forward_cached`
//! - `guards`: preconditions the cached path turns into loud errors
//! - `lora`: adapter attachment and write-back
//!
//! [`attention_core_masked`]: crate::model::attention_core::attention_core_masked

mod block;
mod cached;
mod guards;
mod lora;

pub use block::MiniCpm4Attention;
