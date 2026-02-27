//! HuggingFace config.json format and config loading utilities.

use super::attention::{AttentionConfig, RopeScalingConfig};
use super::universal::{UniversalConfig, default_rms_norm_eps};
use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// HuggingFace config.json format
///
/// Use `to_universal()` to convert to our UniversalConfig format.
#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceConfig {
    #[serde(default)]
    pub model_type: Option<String>,

    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    pub vocab_size: usize,
    pub hidden_size: usize,

    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,

    #[serde(alias = "max_position_embeddings")]
    pub max_seq_len: usize,

    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    #[serde(default, alias = "num_key_value_heads")]
    pub num_kv_heads: Option<usize>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub intermediate_size: Option<usize>,

    #[serde(default = "default_hf_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub sliding_window: Option<usize>,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    #[serde(default)]
    pub rope_scaling: Option<HuggingFaceRopeScaling>,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_hf_rope_theta() -> f32 {
    10000.0
}

/// HuggingFace RoPE scaling format
#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceRopeScaling {
    #[serde(rename = "type", alias = "rope_type")]
    pub scaling_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
}

impl HuggingFaceConfig {
    pub fn from_json(content: &str) -> Result<Self> {
        serde_json::from_str(content).map_err(|e| Error::ModelError {
            reason: format!("Failed to parse HuggingFace config: {e}"),
        })
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        Self::from_json(&content)
    }

    /// Convert to UniversalConfig
    pub fn to_universal(&self) -> UniversalConfig {
        let model_type = self.infer_model_type();

        let attention = self.num_attention_heads.map(|num_heads| {
            let rope_scaling = self.rope_scaling.as_ref().and_then(|rs| {
                rs.scaling_type.as_ref().map(|t| RopeScalingConfig {
                    scaling_type: t.clone(),
                    factor: rs.factor.unwrap_or(1.0),
                    original_max_position_embeddings: rs.original_max_position_embeddings,
                    low_freq_factor: rs.low_freq_factor,
                    high_freq_factor: rs.high_freq_factor,
                    attention_factor: rs.attention_factor,
                    beta_fast: rs.beta_fast,
                    beta_slow: rs.beta_slow,
                })
            });

            AttentionConfig {
                num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                rope_theta: self.rope_theta,
                rope_scaling,
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                sliding_window: self.sliding_window,
            }
        });

        UniversalConfig {
            model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            attention,
            ssm: None,
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }

    fn infer_model_type(&self) -> String {
        if let Some(mt) = &self.model_type {
            return mt.clone();
        }
        if let Some(archs) = &self.architectures {
            if let Some(arch) = archs.first() {
                let arch_lower = arch.to_lowercase();
                if arch_lower.contains("llama") {
                    return "llama".to_string();
                } else if arch_lower.contains("mistral") {
                    return "mistral".to_string();
                } else if arch_lower.contains("mamba") {
                    return "mamba2".to_string();
                }
            }
        }
        "llama".to_string()
    }
}

/// Load config, attempting both UniversalConfig and HuggingFace formats
pub fn load_config_auto<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    // Try UniversalConfig first (our native format)
    if let Ok(config) = serde_json::from_str::<UniversalConfig>(&content) {
        if config.validate().is_ok() {
            return Ok(config);
        }
    }

    // Try YAML format
    if let Ok(config) = serde_yaml::from_str::<UniversalConfig>(&content) {
        if config.validate().is_ok() {
            return Ok(config);
        }
    }

    // Try HuggingFace format
    if let Ok(hf_config) = HuggingFaceConfig::from_json(&content) {
        let config = hf_config.to_universal();
        config.validate()?;
        return Ok(config);
    }

    Err(Error::ModelError {
        reason: "Failed to parse config as UniversalConfig, YAML, or HuggingFace format".into(),
    })
}

/// Load HuggingFace config.json and convert to UniversalConfig
pub fn load_huggingface_config<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let hf_config = HuggingFaceConfig::load(path)?;
    let config = hf_config.to_universal();
    config.validate()?;
    Ok(config)
}
