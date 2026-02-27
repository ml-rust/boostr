//! Universal model configuration.

use super::attention::AttentionConfig;
use super::hybrid::HybridConfig;
use super::moe::MoeConfig;
use super::ssm::SsmConfig;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Universal model configuration
///
/// This struct can represent any model architecture through optional
/// sub-configurations for attention, SSM, MoE, and hybrid layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalConfig {
    /// Model architecture type (e.g., "llama", "mistral", "mamba2", "hybrid")
    pub model_type: String,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension
    pub hidden_size: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// FFN intermediate size (for MLP layers)
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// RMSNorm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Attention configuration (optional for pure SSM)
    #[serde(default)]
    pub attention: Option<AttentionConfig>,

    /// SSM configuration (optional for pure transformer)
    #[serde(default)]
    pub ssm: Option<SsmConfig>,

    /// MoE configuration (optional)
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    /// Hybrid layer assignment (optional)
    #[serde(default)]
    pub hybrid_layers: Option<HybridConfig>,

    /// Whether to tie word embeddings (share embed_tokens and lm_head weights)
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

pub(crate) fn default_rms_norm_eps() -> f64 {
    1e-5
}

impl UniversalConfig {
    /// Validate configuration constraints
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
        if let Some(ssm) = &self.ssm {
            ssm.validate(self.hidden_size)?;
        }
        if let Some(moe) = &self.moe {
            moe.validate()?;
        }
        if let Some(hybrid) = &self.hybrid_layers {
            hybrid.validate(self.num_layers)?;
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

/// Type alias for backward compatibility within boostr
pub type ModelConfig = UniversalConfig;

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
        let config: UniversalConfig = serde_yaml::from_str(yaml).unwrap();
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
        let config: UniversalConfig = serde_yaml::from_str(yaml).unwrap();
        config.validate().unwrap();
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert_eq!(config.intermediate_size(), 1024);
        assert!(!config.tie_word_embeddings);

        let attn = config.attention.as_ref().unwrap();
        assert_eq!(attn.kv_heads(), 4);
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
        let config: UniversalConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.validate().is_err()); // 256 % 3 != 0
    }

    #[test]
    fn test_mamba2_config_yaml() {
        let yaml = r#"
model_type: mamba2
vocab_size: 128256
hidden_size: 768
num_layers: 24
max_seq_len: 8192
ssm:
  variant: mamba2
  num_heads: 48
  head_dim: 32
  state_size: 64
  chunk_size: 64
  n_groups: 1
  conv_kernel: 4
  expand: 2
"#;
        let config: UniversalConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model_type, "mamba2");
        assert!(config.ssm.is_some());
        let ssm = config.ssm.as_ref().unwrap();
        assert_eq!(ssm.variant, "mamba2");
        assert_eq!(ssm.num_heads, 48);
        // 768 * 2 = 1536, 48 * 32 = 1536 ✓
        config.validate().unwrap();
    }

    #[test]
    fn test_hybrid_config_yaml() {
        let yaml = r#"
model_type: hybrid
vocab_size: 128354
hidden_size: 384
num_layers: 8
max_seq_len: 512
attention:
  num_heads: 6
  kv_latent_dim: 192
  rope_theta: 10000.0
ssm:
  variant: mamba2
  num_heads: 48
  head_dim: 16
  state_size: 64
  chunk_size: 64
  expand: 2
hybrid_layers:
  ssm_layers: [0, 1, 2, 4, 5, 6]
  attention_layers: [3, 7]
"#;
        let config: UniversalConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model_type, "hybrid");
        assert!(config.hybrid_layers.is_some());
        let hybrid = config.hybrid_layers.as_ref().unwrap();
        assert!(hybrid.is_ssm_layer(0));
        assert!(hybrid.is_attention_layer(3));
        config.validate().unwrap();
    }
}
