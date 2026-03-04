//! RoPE (Rotary Position Embedding) implementations.
//!
//! Split across submodules by variant:
//! - `rope_standard`    — split-half RoPE (LLaMA/Mistral style)
//! - `rope_interleaved` — interleaved RoPE (GPT-NeoX/Qwen style)
//! - `rope_yarn`        — YaRN extended-context RoPE

#[path = "rope_common.rs"]
mod rope_common;

#[path = "rope_standard.rs"]
pub mod rope_standard;

#[path = "rope_interleaved.rs"]
pub mod rope_interleaved;

#[path = "rope_yarn.rs"]
pub mod rope_yarn;

pub use rope_interleaved::apply_rope_interleaved_impl;
pub use rope_standard::apply_rope_impl;
pub use rope_yarn::apply_rope_yarn_impl;
