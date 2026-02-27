//! Attention configuration types.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,

    #[serde(default)]
    pub num_kv_heads: Option<usize>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// KV latent dimension for MLA compression
    #[serde(default)]
    pub kv_latent_dim: Option<usize>,

    /// Query latent dimension for MLA compression
    #[serde(default)]
    pub q_latent_dim: Option<usize>,

    /// Decoupled RoPE dimension for MLA
    #[serde(default)]
    pub d_rope: Option<usize>,

    /// Sliding window size (for Mistral-style attention)
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl AttentionConfig {
    pub fn validate(&self, hidden_size: usize) -> Result<()> {
        if self.num_heads == 0 {
            return Err(Error::ModelError {
                reason: "num_heads must be > 0".into(),
            });
        }
        if hidden_size % self.num_heads != 0 {
            return Err(Error::ModelError {
                reason: format!(
                    "hidden_size ({hidden_size}) must be divisible by num_heads ({})",
                    self.num_heads
                ),
            });
        }
        if let Some(kv) = self.num_kv_heads {
            if self.num_heads % kv != 0 {
                return Err(Error::ModelError {
                    reason: format!(
                        "num_heads ({}) must be divisible by num_kv_heads ({kv})",
                        self.num_heads
                    ),
                });
            }
        }
        Ok(())
    }

    pub fn head_dim(&self, hidden_size: usize) -> usize {
        self.head_dim.unwrap_or(hidden_size / self.num_heads)
    }

    pub fn kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    /// Check if this is MLA (Multi-Head Latent Attention)
    pub fn is_mla(&self) -> bool {
        self.kv_latent_dim.is_some()
    }

    /// Check if this is GQA (Grouped Query Attention)
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads.is_some() && self.num_kv_heads != Some(self.num_heads)
    }
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub factor: f32,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
}
