//! HuggingFace `config.json` wire format: field layout and file loading.

use crate::error::{Error, Result};
use crate::model::config::universal::default_rms_norm_eps;
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

    /// Audio encoder configuration (Ultravox, Qwen2-Audio, Qwen2.5-Omni)
    #[serde(default)]
    pub audio_config: Option<serde_json::Value>,
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
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
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
}
