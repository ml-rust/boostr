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
    /// Compute the dimension of each attention head (`hidden_size / num_attention_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Build an `EncoderConfig` from GGUF metadata keys.
    ///
    /// Reads standard BERT GGUF keys:
    /// - `bert.embedding_length` → `hidden_size`
    /// - `bert.feed_forward_length` → `intermediate_size`
    /// - `bert.attention.head_count` → `num_attention_heads`
    /// - `bert.block_count` → `num_hidden_layers`
    /// - `bert.context_length` → `max_position_embeddings`
    ///
    /// Vocab size is inferred from the `tokenizer.ggml.tokens` array length.
    pub fn from_gguf_metadata(
        metadata: &crate::format::GgufMetadata,
    ) -> crate::error::Result<Self> {
        use crate::error::Error;

        let hidden_size =
            metadata
                .get_u32("bert.embedding_length")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing bert.embedding_length".into(),
                })? as usize;

        let intermediate_size = metadata
            .get_u32("bert.feed_forward_length")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing bert.feed_forward_length".into(),
            })? as usize;

        let num_attention_heads =
            metadata
                .get_u32("bert.attention.head_count")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing bert.attention.head_count".into(),
                })? as usize;

        let num_hidden_layers =
            metadata
                .get_u32("bert.block_count")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing bert.block_count".into(),
                })? as usize;

        let max_position_embeddings =
            metadata.get_u32("bert.context_length").unwrap_or(512) as usize;

        let vocab_size = metadata
            .get_array("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .unwrap_or(30522); // BERT default

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
        })
    }
}
