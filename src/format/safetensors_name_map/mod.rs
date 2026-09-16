//! HuggingFace SafeTensors tensor name normalization.
//!
//! Maps non-standard HF tensor names to canonical Llama-style names at load time.
//! Same pattern as `gguf_to_hf_name()` in the GGUF module.
//!
//! The canonical naming convention is:
//! ```text
//! model.embed_tokens.weight
//! model.layers.{N}.self_attn.{q,k,v,o}_proj.weight
//! model.layers.{N}.mlp.{gate,up,down}_proj.weight
//! model.layers.{N}.input_layernorm.weight
//! model.layers.{N}.post_attention_layernorm.weight
//! model.norm.weight
//! lm_head.weight
//! ```

mod dbrx;
mod dispatch;
mod falcon;
mod gpt_neox;

pub use dispatch::{normalize_hf_name, uses_fused_qkv};
