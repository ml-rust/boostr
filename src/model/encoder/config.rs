//! Encoder model configuration (BERT-style transformer encoders).

use serde::{Deserialize, Serialize};

/// Configuration for BERT-style transformer encoder models.
///
/// Compatible with HuggingFace `config.json` for models like
/// `all-MiniLM-L6-v2`, `bge-small-en`, `nomic-embed-text`, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub hidden_act: HiddenAct,
    #[serde(default)]
    pub type_vocab_size: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    #[default]
    Gelu,
    Relu,
}

fn default_eps() -> f64 {
    1e-12
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
