//! Multi-Head Latent Attention (MLA) module
//!
//! DeepSeek-V2 style attention with low-rank KV compression.
//! Compresses KV cache from O(L * n_heads * head_dim * 2) to O(L * (kv_lora_rank + rope_head_dim)).
//!
//! Architecture:
//! - Q path: optional low-rank compression (q_down → norm → q_up) or direct projection
//! - KV path: compress → split (c_kv, k_pe) → norm c_kv → decompress → split (k_nope, v)
//! - Decoupled RoPE: applied only to q_pe and k_pe portions
//! - Attention: Q=[q_nope, q_pe], K=[k_nope, k_pe], V=v

mod config;
mod model;
mod parameter_identity;

pub use config::MlaConfig;
pub use model::Mla;
pub use parameter_identity::MlaWeights;
