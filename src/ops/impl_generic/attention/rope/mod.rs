//! RoPE (Rotary Position Embedding) implementations.
//!
//! Split across submodules by variant:
//! - `rope_standard`    — split-half RoPE (LLaMA/Mistral style)
//! - `rope_interleaved` — interleaved RoPE (GPT-NeoX/Qwen style)
//! - `rope_yarn`        — YaRN extended-context RoPE

mod common;

pub mod standard;

pub mod interleaved;

pub mod yarn;

pub use interleaved::apply_rope_interleaved_impl;
pub use standard::apply_rope_impl;
pub use yarn::apply_rope_yarn_impl;
