//! HuggingFace `config.json` wire format: field layout and file loading.

use crate::error::{Error, Result};
use crate::model::config::universal::default_rms_norm_eps;
use serde::Deserialize;
use std::path::Path;

/// HuggingFace config.json format
///
/// Use `to_universal()` to convert to our UniversalConfig format.
///
/// Field aliases cover the GPT-NeoX, Falcon and DBRX key names
/// (`n_head`, `n_layer`, `d_model`, ...).
#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceConfig {
    #[serde(default)]
    pub model_type: Option<String>,

    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    pub vocab_size: usize,

    #[serde(alias = "d_model")]
    pub hidden_size: usize,

    #[serde(alias = "num_hidden_layers", alias = "n_layer", alias = "n_layers")]
    pub num_layers: usize,

    /// Falcon-RW configs omit it. `FalconConfig` defaults it to 2048.
    #[serde(
        alias = "max_position_embeddings",
        alias = "n_positions",
        default = "default_max_positions"
    )]
    pub max_seq_len: usize,

    #[serde(default, alias = "n_head", alias = "n_heads")]
    pub num_attention_heads: Option<usize>,

    #[serde(default, alias = "num_key_value_heads")]
    pub num_kv_heads: Option<usize>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// RoPE base. GPT-NeoX names it `rotary_emb_base`. A transformers v5
    /// file carries it in `rope_parameters` instead, and DBRX in
    /// `attn_config`.
    #[serde(default, alias = "rotary_emb_base")]
    pub rope_theta: Option<f32>,

    /// transformers v5 RoPE block: `{"rope_theta": .., "rope_type": ..}`.
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,

    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Qwen2 ships `sliding_window` with `use_sliding_window: false`.
    #[serde(default)]
    pub use_sliding_window: Option<bool>,

    /// Qwen2: layers from this index on slide when `use_sliding_window`.
    #[serde(default)]
    pub max_window_layers: Option<usize>,

    /// Per-layer attention kind: `sliding_attention` or `full_attention`.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// LayerNorm epsilon of the LayerNorm architectures.
    #[serde(default, alias = "layer_norm_epsilon", alias = "norm_epsilon")]
    pub layer_norm_eps: Option<f64>,

    #[serde(default)]
    pub rope_scaling: Option<HuggingFaceRopeScaling>,

    /// Pretraining context of a LongRoPE model (Phi-3). Phi-3 configs set
    /// it here and leave it out of `rope_scaling`.
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,

    /// `None` takes the architecture's own default: true for Gemma,
    /// Cohere, Falcon and StarCoder2, false otherwise.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    // ── MoE fields ──────────────────────────────────────────────────
    /// Number of experts (Mixtral; Qwen2-MoE names it `num_experts`)
    #[serde(default, alias = "num_experts")]
    pub num_local_experts: Option<usize>,

    /// Number of active experts per token (top-k routing)
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    /// Qwen2-MoE: renormalize the top-k routing weights.
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,

    /// Qwen2-MoE expert width.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

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

    /// Parallel attention + MLP with one norm (Falcon)
    #[serde(default)]
    pub parallel_attn: Option<bool>,

    /// Parallel attention + MLP with two norms (GPT-NeoX)
    #[serde(default)]
    pub use_parallel_residual: Option<bool>,

    /// MLP activation. Gemma names it `hidden_activation`.
    #[serde(default)]
    pub hidden_act: Option<String>,

    #[serde(default)]
    pub hidden_activation: Option<String>,

    /// Rotated fraction of each head (Phi, StarCoder2 forks).
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,

    /// Rotated fraction of each head (GPT-NeoX).
    #[serde(default)]
    pub rotary_pct: Option<f32>,

    /// Gemma 2: attention scale is `query_pre_attn_scalar^-0.5`.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,

    /// Gemma 2 attention score softcap.
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,

    /// Gemma 2 output logit softcap.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,

    /// Cohere output logit multiplier.
    #[serde(default)]
    pub logit_scale: Option<f32>,

    /// Cohere per-head Q/K LayerNorm.
    #[serde(default)]
    pub use_qk_norm: Option<bool>,

    /// DBRX attention block: `kv_n_heads`, `clip_qkv`, `rope_theta`.
    #[serde(default)]
    pub attn_config: Option<serde_json::Value>,

    /// DBRX FFN block: `moe_num_experts`, `moe_top_k`, `ffn_hidden_size`,
    /// `ffn_act_fn`, `moe_normalize_expert_weights`.
    #[serde(default)]
    pub ffn_config: Option<serde_json::Value>,

    // ── Vision/multimodal fields ─────────────────────────────────────
    /// Vision encoder configuration (LLaVA, Qwen-VL, etc.)
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,

    /// Audio encoder configuration (Ultravox, Qwen2-Audio, Qwen2.5-Omni)
    #[serde(default)]
    pub audio_config: Option<serde_json::Value>,
}

fn default_max_positions() -> usize {
    2048
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
