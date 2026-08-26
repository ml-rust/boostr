//! VoxCPM2's MiniCPM4 decoder-only transformer: causal forward over
//! pre-computed embeddings, full-sequence or KV-cached, no `lm_head`.
//! Instantiated twice — `base_lm` and the 8-layer NoPE `residual_lm`.

pub mod attention;
pub mod config;
pub mod decode;
pub mod layer;
pub mod loader;
pub mod mlp;
pub mod model;

pub use attention::MiniCpm4Attention;
pub use config::{
    DEFAULT_CONFIG_SECTION, MiniCpm4Config, RESIDUAL_LM_NO_ROPE_KEY, RESIDUAL_LM_NUM_LAYERS_KEY,
};
pub use layer::MiniCpm4Layer;
pub use loader::{DEFAULT_MINICPM4_PREFIX, DEFAULT_RESIDUAL_LM_PREFIX};
pub use mlp::MiniCpm4Mlp;
pub use model::MiniCpm4Model;
