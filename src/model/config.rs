//! Model configuration system
//!
//! Supports YAML and JSON formats, including HuggingFace config.json.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Universal model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,

    #[serde(default)]
    pub intermediate_size: Option<usize>,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default)]
    pub attention: Option<AttentionConfig>,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(Error::ModelError {
                reason: "vocab_size must be > 0".into(),
            });
        }
        if self.hidden_size == 0 {
            return Err(Error::ModelError {
                reason: "hidden_size must be > 0".into(),
            });
        }
        if self.num_layers == 0 {
            return Err(Error::ModelError {
                reason: "num_layers must be > 0".into(),
            });
        }
        if let Some(attn) = &self.attention {
            attn.validate(self.hidden_size)?;
        }
        Ok(())
    }

    pub fn load_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let config: Self = serde_yaml::from_str(&content).map_err(|e| Error::ModelError {
            reason: format!("YAML parse error: {e}"),
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let config: Self = serde_json::from_str(&content).map_err(|e| Error::ModelError {
            reason: format!("JSON parse error: {e}"),
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Get the intermediate (FFN) size, defaulting to 4 * hidden_size
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }
}

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_config_yaml() {
        let yaml = r#"
model_type: llama
vocab_size: 128256
hidden_size: 4096
num_layers: 32
max_seq_len: 8192
intermediate_size: 14336
attention:
  num_heads: 32
  num_kv_heads: 8
  rope_theta: 500000.0
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.vocab_size, 128256);
        config.validate().unwrap();

        let attn = config.attention.as_ref().unwrap();
        assert_eq!(attn.num_heads, 32);
        assert_eq!(attn.kv_heads(), 8);
        assert_eq!(attn.head_dim(4096), 128);
    }

    #[test]
    fn test_config_defaults() {
        let yaml = r#"
model_type: llama
vocab_size: 1000
hidden_size: 256
num_layers: 4
max_seq_len: 512
attention:
  num_heads: 4
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        config.validate().unwrap();
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert_eq!(config.intermediate_size(), 1024);
        assert!(!config.tie_word_embeddings);

        let attn = config.attention.as_ref().unwrap();
        assert_eq!(attn.kv_heads(), 4); // defaults to num_heads
        assert_eq!(attn.rope_theta, 10000.0);
    }

    #[test]
    fn test_config_validation_bad_heads() {
        let yaml = r#"
model_type: llama
vocab_size: 1000
hidden_size: 256
num_layers: 4
max_seq_len: 512
attention:
  num_heads: 3
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.validate().is_err()); // 256 % 3 != 0
    }
}
