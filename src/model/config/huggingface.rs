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
                // Order matters: check specific variants before generic ones
                // (e.g. "qwen2moe" before "qwen", "phi3" before "phi",
                //  "gemma2" before "gemma").
                if arch_lower.contains("llama") {
                    return "llama".to_string();
                } else if arch_lower.contains("mistral") {
                    return "mistral".to_string();
                } else if arch_lower.contains("mamba") {
                    return "mamba2".to_string();
                } else if arch_lower.contains("qwen2moe") {
                    return "qwen2_moe".to_string();
                } else if arch_lower.contains("qwen") {
                    return "qwen2".to_string();
                } else if arch_lower.contains("phi3") {
                    return "phi3".to_string();
                } else if arch_lower.contains("phi") {
                    return "phi".to_string();
                } else if arch_lower.contains("gemma2") {
                    return "gemma2".to_string();
                } else if arch_lower.contains("gemma") {
                    return "gemma".to_string();
                } else if arch_lower.contains("starcoder") {
                    return "starcoder2".to_string();
                } else if arch_lower.contains("internlm") {
                    return "internlm2".to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal HuggingFaceConfig with only architectures set
    /// (model_type = None) to exercise the fallback inference path.
    fn config_with_arch(arch: &str) -> HuggingFaceConfig {
        HuggingFaceConfig {
            model_type: None,
            architectures: Some(vec![arch.to_string()]),
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            max_seq_len: 4096,
            num_attention_heads: Some(32),
            num_kv_heads: None,
            head_dim: None,
            intermediate_size: None,
            rope_theta: 10000.0,
            sliding_window: None,
            rms_norm_eps: 1e-5,
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    /// Helper: build a minimal HuggingFaceConfig with model_type set directly
    /// (the primary path — HF configs almost always have model_type).
    fn config_with_model_type(mt: &str) -> HuggingFaceConfig {
        let mut c = config_with_arch("Unused");
        c.model_type = Some(mt.to_string());
        c
    }

    // -- model_type passthrough (primary path) --

    #[test]
    fn model_type_passthrough() {
        // When model_type is present, infer_model_type returns it verbatim.
        for mt in &[
            "llama",
            "mistral",
            "qwen2",
            "qwen2_moe",
            "phi3",
            "phi",
            "gemma",
            "gemma2",
            "starcoder2",
            "internlm2",
        ] {
            let c = config_with_model_type(mt);
            assert_eq!(c.infer_model_type(), *mt, "passthrough failed for {mt}");
        }
    }

    // -- architecture fallback (when model_type is absent) --
    // Uses real HuggingFace architectures[0] values.

    #[test]
    fn arch_fallback_llama() {
        // LlamaForCausalLM covers Llama, CodeLlama, Yi, Solar
        assert_eq!(
            config_with_arch("LlamaForCausalLM").infer_model_type(),
            "llama"
        );
    }

    #[test]
    fn arch_fallback_mistral() {
        assert_eq!(
            config_with_arch("MistralForCausalLM").infer_model_type(),
            "mistral"
        );
    }

    #[test]
    fn arch_fallback_qwen2() {
        assert_eq!(
            config_with_arch("Qwen2ForCausalLM").infer_model_type(),
            "qwen2"
        );
    }

    #[test]
    fn arch_fallback_qwen2_moe() {
        // Qwen2MoeForCausalLM must map to "qwen2_moe", NOT "qwen2"
        assert_eq!(
            config_with_arch("Qwen2MoeForCausalLM").infer_model_type(),
            "qwen2_moe"
        );
    }

    #[test]
    fn arch_fallback_phi3() {
        assert_eq!(
            config_with_arch("Phi3ForCausalLM").infer_model_type(),
            "phi3"
        );
    }

    #[test]
    fn arch_fallback_phi() {
        // Phi-2 uses "PhiForCausalLM", must map to "phi" not "phi3"
        assert_eq!(config_with_arch("PhiForCausalLM").infer_model_type(), "phi");
    }

    #[test]
    fn arch_fallback_gemma2() {
        assert_eq!(
            config_with_arch("Gemma2ForCausalLM").infer_model_type(),
            "gemma2"
        );
    }

    #[test]
    fn arch_fallback_gemma() {
        // Gemma (v1) must map to "gemma", not "gemma2"
        assert_eq!(
            config_with_arch("GemmaForCausalLM").infer_model_type(),
            "gemma"
        );
    }

    #[test]
    fn arch_fallback_starcoder2() {
        assert_eq!(
            config_with_arch("Starcoder2ForCausalLM").infer_model_type(),
            "starcoder2"
        );
    }

    #[test]
    fn arch_fallback_internlm2() {
        assert_eq!(
            config_with_arch("InternLM2ForCausalLM").infer_model_type(),
            "internlm2"
        );
    }

    #[test]
    fn arch_fallback_mamba() {
        assert_eq!(
            config_with_arch("MambaForCausalLM").infer_model_type(),
            "mamba2"
        );
    }

    #[test]
    fn arch_fallback_unknown_defaults_to_llama() {
        assert_eq!(
            config_with_arch("SomeNewModelForCausalLM").infer_model_type(),
            "llama"
        );
    }
}

/// Load HuggingFace config.json and convert to UniversalConfig
pub fn load_huggingface_config<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let hf_config = HuggingFaceConfig::load(path)?;
    let config = hf_config.to_universal();
    config.validate()?;
    Ok(config)
}
