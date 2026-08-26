//! VoxCPM2's MiniCPM4 decoder-only transformer (`base_lm`): full-sequence
//! causal forward over pre-computed embeddings, no KV cache, no `lm_head`.

pub mod attention;
pub mod config;
pub mod layer;
pub mod loader;
pub mod mlp;
pub mod model;

pub use attention::MiniCpm4Attention;
pub use config::{DEFAULT_CONFIG_SECTION, MiniCpm4Config};
pub use layer::MiniCpm4Layer;
pub use loader::DEFAULT_MINICPM4_PREFIX;
pub use mlp::MiniCpm4Mlp;
pub use model::MiniCpm4Model;
