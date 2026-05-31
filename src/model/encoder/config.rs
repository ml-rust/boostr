//! Encoder model configuration (BERT-style transformer encoders).

use numr::dtype::DType;
use serde::{Deserialize, Serialize};

/// Architecture family for position-id generation and embedding behaviour.
///
/// BERT uses simple 0-based position ids.  XLM-RoBERTa (used by e.g.
/// bge-reranker-v2-m3) reserves position `pad_token_id` for padding and
/// numbers real tokens starting from `pad_token_id + 1`.
/// NomicBert replaces learned position embeddings with RoPE and uses a
/// SwiGLU FFN.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArchFamily {
    /// Standard BERT: position_ids = [0, 1, ..., S-1].
    #[default]
    Bert,
    /// XLM-RoBERTa: position_ids computed as cumsum(input_ids != pad_id) + pad_id,
    /// with padding positions assigned position_id = pad_id.
    XlmRoberta,
    /// NomicBert: RoPE positions (no learned position embedding), SwiGLU FFN,
    /// fused QKV projection, token-type embedding row 0, mean pooling.
    NomicBert,
    /// Gemma3-embedding: RoPE positions, sandwich RMSNorm, GQA, QK-norm,
    /// GeGLU FFN, token-embedding scale sqrt(hidden_size), mean pooling.
    /// No learned position embedding. No biases anywhere.
    GemmaEmbedding,
}

/// FFN variant: controls which feed-forward computation is used per layer.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FfnVariant {
    /// Standard BERT FFN: ffn_down(act(ffn_up(x))).
    #[default]
    Standard,
    /// NomicBert SwiGLU: ffn_down(silu(ffn_gate(x)) * ffn_up(x)).
    GatedSilu,
    /// Gemma GeGLU: ffn_down(gelu(ffn_gate(x)) * ffn_up(x)).
    /// Gate activation is GELU (not SiLU/SwiGLU).
    GatedGelu,
}

/// Maximum number of packed tokens per varlen forward pass.
///
/// Bounds peak memory for `embed_texts_varlen` by splitting large document
/// batches into sub-batches whose total token count does not exceed this
/// value.  A single document that is longer than this limit is still
/// processed in one forward (documents cannot be split).
///
/// Tuned per hardware: 16 384 tokens fits comfortably on most 24 GB GPUs
/// when hidden_size ≤ 768 and up to 12 layers.  Reduce to 8 192 on
/// smaller GPUs or larger models.
pub const DEFAULT_MAX_TOKENS_PER_FORWARD: usize = 16384;

/// Configuration for BERT-style transformer encoder models.
///
/// Compatible with HuggingFace `config.json` for models like
/// `all-MiniLM-L6-v2`, `bge-small-en`, `nomic-embed-text`, etc.
/// Also supports XLM-RoBERTa backbones (bge-reranker-v2-m3, etc.)
/// via the `arch_family` field.
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
    /// Architecture family — controls position-id generation strategy.
    #[serde(default)]
    pub arch_family: ArchFamily,
    /// Pad token ID — used by XLM-RoBERTa position-id computation.
    /// BERT default: 0.  XLM-RoBERTa default: 1.
    #[serde(default)]
    pub padding_token_id: i64,
    /// Compute dtype for the encoder forward pass.
    ///
    /// `DType::F32` (the default) reproduces existing behaviour exactly.
    /// `DType::F16` pre-dequantizes quantized projection weights to F16 at load
    /// time and runs all activations in F16, routing through numr's WMMA tensor-
    /// core GEMM and fused F16 kernels.  The pooled output is always cast back to
    /// F32 before returning so callers (e.g. the classifier head and CUDA graph
    /// buffers) remain unchanged.
    ///
    /// Only effective on CUDA; ignored on CPU (weights and activations stay F32).
    #[serde(skip, default = "default_compute_dtype")]
    pub compute_dtype: DType,

    // ── NomicBert-only fields (serde-defaulted; BERT/XLM-R deserialization unchanged) ──
    /// RoPE frequency base. Read from `nomic-bert.rope.freq_base`. Default 10000.0.
    #[serde(default = "default_rope_freq_base")]
    pub rope_freq_base: f32,

    /// Whether the model uses causal (autoregressive) attention.
    /// NomicBert is bidirectional; always `false` in practice. Default false.
    #[serde(default)]
    pub causal: bool,

    /// FFN variant: Standard (BERT) or GatedSilu (NomicBert SwiGLU).
    #[serde(default)]
    pub ffn_variant: FfnVariant,

    /// Size of the token-type embedding table (`type_vocab_size` for NomicBert).
    /// Zero for BERT models that do not load token_types.weight.
    #[serde(default)]
    pub token_type_embed_size: usize,

    // ── Gemma-embedding-only fields (serde-defaulted; existing archs unchanged) ──
    /// Number of KV heads (GQA). Equal to `num_attention_heads` for MHA (BERT/NomicBert).
    /// Read from `gemma-embedding.attention.head_count_kv`. Default 0 → falls back to
    /// `num_attention_heads` via `resolved_num_kv_heads()`.
    #[serde(default)]
    pub num_kv_heads: usize,

    /// Explicit per-head dimension from `gemma-embedding.attention.key_length`.
    /// When `None`, resolved as `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim_explicit: Option<usize>,

    /// RMSNorm epsilon from `gemma-embedding.attention.layer_norm_rms_epsilon`.
    /// Also stored in `layer_norm_eps` for the norm forward; this field records
    /// the GGUF value separately when the general `layer_norm_eps` field is already
    /// used by the BERT path. Default 1e-6.
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f64,

    /// Sliding window size from `gemma-embedding.attention.sliding_window`.
    /// Stored but not enforced for our embedding use (sessions ≤ 128 tokens are
    /// well within the window). Full bidirectional attention is always applied.
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// When true, token embeddings are multiplied by sqrt(hidden_size) after lookup.
    /// Required for Gemma correctness; not a tensor — pure scalar multiply.
    #[serde(default)]
    pub embed_scale: bool,

    /// Maximum number of packed tokens per varlen forward pass.
    ///
    /// `None` resolves to [`DEFAULT_MAX_TOKENS_PER_FORWARD`] (16 384).
    /// Set explicitly to tune peak GPU memory vs. throughput trade-off.
    /// Only affects the NomicBert varlen path (`embed_texts_varlen`).
    #[serde(default)]
    pub max_tokens_per_forward: Option<usize>,
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

fn default_compute_dtype() -> DType {
    DType::F32
}

fn default_rope_freq_base() -> f32 {
    10000.0
}

fn default_rms_eps() -> f64 {
    1e-6
}

impl EncoderConfig {
    /// Compute the dimension of each attention head (`hidden_size / num_attention_heads`).
    ///
    /// This is the BERT/NomicBert formula. For Gemma, use `resolved_head_dim()`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Resolve the per-head dimension.
    ///
    /// Returns `head_dim_explicit` when set (Gemma stores it as
    /// `gemma-embedding.attention.key_length`). Falls back to the BERT formula
    /// `hidden_size / num_attention_heads` when `None`.
    pub fn resolved_head_dim(&self) -> usize {
        self.head_dim_explicit
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    /// Resolve the number of KV heads.
    ///
    /// Returns the stored `num_kv_heads` when non-zero (Gemma GQA).
    /// Falls back to `num_attention_heads` (MHA for BERT/NomicBert).
    pub fn resolved_num_kv_heads(&self) -> usize {
        if self.num_kv_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_kv_heads
        }
    }
}

#[path = "config_gguf.rs"]
mod gguf;
