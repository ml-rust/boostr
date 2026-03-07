//! HuggingFace config.json format and config loading utilities.

use super::attention::{AttentionConfig, RopeScalingConfig};
use super::universal::{UniversalConfig, default_rms_norm_eps};
use super::vision::VisionConfig;
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

    // ── MoE fields ──────────────────────────────────────────────────
    /// Number of experts (Mixtral, Qwen2-MoE, DBRX)
    #[serde(default)]
    pub num_local_experts: Option<usize>,

    /// Number of active experts per token (top-k routing)
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    // ── Architecture-specific flags ─────────────────────────────────
    /// Whether attention layers use bias (GPT-NeoX, Falcon, some Qwen)
    #[serde(default)]
    pub attention_bias: Option<bool>,

    /// Use ALiBi position embeddings instead of RoPE (Falcon v1)
    #[serde(default)]
    pub alibi: Option<bool>,

    /// Multi-query attention flag (Falcon)
    #[serde(default)]
    pub multi_query: Option<bool>,

    /// New decoder architecture flag (Falcon-40B+)
    #[serde(default)]
    pub new_decoder_architecture: Option<bool>,

    /// Parallel attention + MLP (GPT-NeoX)
    #[serde(default)]
    pub parallel_attn: Option<bool>,

    // ── Vision/multimodal fields ─────────────────────────────────────
    /// Vision encoder configuration (LLaVA, Qwen-VL, etc.)
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
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

        // Determine num_kv_heads: Falcon multi_query means 1 KV head
        let effective_kv_heads = if self.multi_query == Some(true) && self.num_kv_heads.is_none() {
            Some(1)
        } else {
            self.num_kv_heads
        };

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
                num_kv_heads: effective_kv_heads,
                head_dim: self.head_dim,
                rope_theta: self.rope_theta,
                rope_scaling,
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                sliding_window: self.sliding_window,
                use_alibi: self.alibi.unwrap_or(false),
            }
        });

        // Auto-detect MoE from num_local_experts
        let moe = self
            .num_local_experts
            .map(|num_experts| super::moe::MoeConfig {
                num_experts,
                experts_per_tok: self.num_experts_per_tok.unwrap_or(2),
                shared_expert: None,
                intermediate_size: None,
                load_balance_alpha: 0.01,
                z_loss_alpha: 1e-3,
            });

        // Parse vision_config JSON into VisionConfig if present
        let vision = self
            .vision_config
            .as_ref()
            .and_then(|vc| serde_json::from_value::<VisionConfig>(vc.clone()).ok());

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
            moe,
            hybrid_layers: None,
            tie_word_embeddings: self.tie_word_embeddings,
            vision,
            audio: None,
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
                if arch_lower.contains("llava") {
                    return "llava".to_string();
                } else if arch_lower.contains("qwen_vl") || arch_lower.contains("qwenvl") {
                    return "qwen_vl".to_string();
                } else if arch_lower.contains("llama") {
                    return "llama".to_string();
                } else if arch_lower.contains("mixtral") {
                    return "mixtral".to_string();
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
                } else if arch_lower.contains("falcon") {
                    return "falcon".to_string();
                } else if arch_lower.contains("neox") || arch_lower.contains("pythia") {
                    return "gpt_neox".to_string();
                } else if arch_lower.contains("dbrx") {
                    return "dbrx".to_string();
                } else if arch_lower.contains("cohere") || arch_lower.contains("command") {
                    return "command_r".to_string();
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
            num_local_experts: None,
            num_experts_per_tok: None,
            attention_bias: None,
            alibi: None,
            multi_query: None,
            new_decoder_architecture: None,
            parallel_attn: None,
            vision_config: None,
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

    #[test]
    fn arch_fallback_falcon() {
        assert_eq!(
            config_with_arch("FalconForCausalLM").infer_model_type(),
            "falcon"
        );
    }

    #[test]
    fn arch_fallback_gpt_neox() {
        assert_eq!(
            config_with_arch("GPTNeoXForCausalLM").infer_model_type(),
            "gpt_neox"
        );
    }

    #[test]
    fn arch_fallback_dbrx() {
        assert_eq!(
            config_with_arch("DbrxForCausalLM").infer_model_type(),
            "dbrx"
        );
    }

    #[test]
    fn arch_fallback_mixtral() {
        assert_eq!(
            config_with_arch("MixtralForCausalLM").infer_model_type(),
            "mixtral"
        );
    }

    #[test]
    fn arch_fallback_command_r() {
        assert_eq!(
            config_with_arch("CohereForCausalLM").infer_model_type(),
            "command_r"
        );
    }

    // -- MoE auto-detection from HF fields --

    #[test]
    fn moe_auto_detection_from_num_local_experts() {
        let mut c = config_with_model_type("mixtral");
        c.num_local_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        let uc = c.to_universal();
        let moe = uc.moe.as_ref().expect("MoE config should be populated");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.experts_per_tok, 2);
    }

    #[test]
    fn no_moe_when_field_absent() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        assert!(uc.moe.is_none());
    }

    #[test]
    fn multi_query_sets_one_kv_head() {
        let mut c = config_with_model_type("falcon");
        c.multi_query = Some(true);
        let uc = c.to_universal();
        let attn = uc.attention.as_ref().unwrap();
        assert_eq!(attn.num_kv_heads, Some(1));
    }

    #[test]
    fn alibi_propagated_to_universal() {
        let mut c = config_with_model_type("falcon");
        c.alibi = Some(true);
        let uc = c.to_universal();
        let attn = uc.attention.as_ref().unwrap();
        assert!(attn.use_alibi);
    }

    #[test]
    fn arch_fallback_llava() {
        assert_eq!(
            config_with_arch("LlavaForConditionalGeneration").infer_model_type(),
            "llava"
        );
    }

    #[test]
    fn arch_fallback_qwen_vl() {
        assert_eq!(
            config_with_arch("QwenVLForConditionalGeneration").infer_model_type(),
            "qwen_vl"
        );
    }

    #[test]
    fn vision_config_parsed_from_hf() {
        let json = r#"{
            "model_type": "llava",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "num_attention_heads": 32,
            "vision_config": {
                "encoder_type": "clip",
                "hidden_size": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "patch_size": 14,
                "image_size": 336,
                "intermediate_size": 4096
            }
        }"#;
        let hf: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        let uc = hf.to_universal();
        assert_eq!(uc.model_type, "llava");
        let vision = uc.vision.as_ref().expect("vision config should be parsed");
        assert_eq!(vision.encoder_type, "clip");
        assert_eq!(vision.hidden_size, 1024);
        assert_eq!(vision.patch_size, 14);
        assert_eq!(vision.projector_type, "linear"); // default
    }

    #[test]
    fn no_vision_when_absent() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        assert!(uc.vision.is_none());
        assert!(uc.audio.is_none());
    }

    #[test]
    fn alibi_defaults_to_false() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        let attn = uc.attention.as_ref().unwrap();
        assert!(!attn.use_alibi);
    }
}
