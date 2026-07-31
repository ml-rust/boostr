//! [`EncoderConfig`] — the full configuration for a transformer encoder.

use super::{ArchFamily, FfnVariant, HiddenAct, LayerAttention, NormScheme};
use numr::dtype::DType;
use serde::{Deserialize, Serialize};

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

/// Default RoPE base for local (windowed) blocks of an interleaved
/// architecture when the model file carries no explicit override.
///
/// Matches llama.cpp's `llama_hparams::rope_freq_base_train_swa`, which is
/// initialised to 10 000 and left untouched when the GGUF omits
/// `*.rope.freq_base_swa` — it is *not* backfilled from the global base.
pub const DEFAULT_LOCAL_ROPE_FREQ_BASE: f32 = 10_000.0;

/// Default local/global block period for Gemma3-derived architectures when the
/// model file omits `*.attention.sliding_window_pattern`.
///
/// Matches llama.cpp's `swa_period = 6` default for `gemma-embedding`.
pub const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;

/// Configuration for transformer encoder models.
///
/// Compatible with HuggingFace `config.json` for models like
/// `all-MiniLM-L6-v2`, `bge-small-en`, `nomic-embed-text`, etc.
/// Also supports XLM-RoBERTa backbones (bge-reranker-v2-m3, etc.),
/// Gemma3-embedding and Qwen3-embedding via the `arch_family` field.
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

    /// RoPE frequency base for global (full-attention) blocks.
    /// Read from `<arch>.rope.freq_base`. Default 10 000.
    #[serde(default = "default_rope_freq_base")]
    pub rope_freq_base: f32,

    /// RoPE frequency base for local (windowed) blocks of an interleaved
    /// architecture. Read from `<arch>.rope.freq_base_swa`; when that key is
    /// absent it stays at [`DEFAULT_LOCAL_ROPE_FREQ_BASE`] rather than
    /// following `rope_freq_base`.
    ///
    /// Only consulted when the architecture interleaves — see
    /// [`EncoderConfig::layer_attention`].
    #[serde(default = "default_local_rope_freq_base")]
    pub rope_freq_base_local: f32,

    /// Whether the model uses causal (autoregressive) attention.
    /// BERT/NomicBert/Gemma-embedding are bidirectional; Qwen3-embedding is causal.
    #[serde(default)]
    pub causal: bool,

    /// FFN variant: Standard (BERT), GatedSilu (NomicBert/Qwen3), GatedGelu (Gemma).
    #[serde(default)]
    pub ffn_variant: FfnVariant,

    /// Where normalisation sits relative to the residual add.
    #[serde(default)]
    pub norm_scheme: NormScheme,

    /// Size of the token-type embedding table (`type_vocab_size` for NomicBert).
    /// Zero for BERT models that do not load token_types.weight.
    #[serde(default)]
    pub token_type_embed_size: usize,

    /// Number of KV heads (GQA). Equal to `num_attention_heads` for MHA.
    /// Default 0 → falls back to `num_attention_heads` via `resolved_num_kv_heads()`.
    #[serde(default)]
    pub num_kv_heads: usize,

    /// Explicit per-head dimension from `<arch>.attention.key_length`.
    /// When `None`, resolved as `hidden_size / num_attention_heads`.
    ///
    /// Must not be derived for Qwen3-embedding, where `key_length` is 128 while
    /// `hidden_size / head_count` is 64.
    #[serde(default)]
    pub head_dim_explicit: Option<usize>,

    /// RMSNorm epsilon from `<arch>.attention.layer_norm_rms_epsilon`.
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f64,

    /// Sliding window size, in positions, from
    /// `<arch>.attention.sliding_window`.
    ///
    /// `Some(w)` marks an interleaved architecture: local blocks attend within a
    /// *symmetric* window of `w` positions (half-width `w / 2`) and rotate at
    /// `rope_freq_base_local`; global blocks attend fully and rotate at
    /// `rope_freq_base`. `None` means every block is global.
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Block period of the local/global alternation. Blocks with
    /// `il % pattern < pattern - 1` are local, the rest global — so with the
    /// default period of 6 and 24 blocks, blocks 5, 11, 17 and 23 are global.
    ///
    /// `0` disables interleaving regardless of `sliding_window`.
    #[serde(default)]
    pub sliding_window_pattern: usize,

    /// How many leading rows were already removed from the learned position
    /// table by whoever produced the weights.
    ///
    /// XLM-RoBERTa numbers real tokens from `pad_token_id + 1`, so its
    /// HuggingFace table carries `pad_token_id + 1` dead rows at the front.
    /// llama.cpp's converter chops them (`XLMRobertaModel` in
    /// `convert_hf_to_gguf.py`), which is why `bge-m3` ships an 8192-row table
    /// for an 8194-position model — so a GGUF-sourced XLM-R must look up
    /// *0-based* positions, while the same model loaded from SafeTensors must
    /// still add the offset.
    ///
    /// Getting this wrong reads every token's position embedding two slots
    /// late. Nothing about the shapes objects (the table is longer than any
    /// real sequence) and the model keeps working — measured against llama.cpp
    /// on bge-m3, the sentence embedding sat at cosine 0.989 instead of 0.999.
    #[serde(default)]
    pub position_embd_offset: i64,

    /// Read-out strategy the model file itself declares, as the raw GGUF
    /// `<arch>.pooling_type` code (1 = mean, 2 = CLS, 3 = last).
    ///
    /// `None` means the file said nothing and the architecture default applies.
    /// This is not decorative: `bge-m3` ships a `bert`-namespace GGUF declaring
    /// `pooling_type = 2`, and mean-pooling it instead returns a usable-looking
    /// vector that is simply not the one the model was trained to produce
    /// (measured against llama.cpp on the same file: cosine 0.74).
    #[serde(default)]
    pub declared_pooling_type: Option<u32>,

    /// Maximum ALiBi bias, when this architecture encodes position as a
    /// per-head linear penalty on key distance rather than with RoPE or a
    /// learned table.
    ///
    /// `Some(8.0)` for jina-bert-v2, matching llama.cpp's
    /// `hparams.f_max_alibi_bias`. `None` — the default — means no ALiBi.
    /// See [`super::alibi::alibi_slopes`] for how the per-head slopes are
    /// derived from this value.
    #[serde(default)]
    pub alibi_max_bias: Option<f32>,

    /// When true, token embeddings are multiplied by sqrt(hidden_size) after lookup.
    /// Required for Gemma correctness; not a tensor — pure scalar multiply.
    #[serde(default)]
    pub embed_scale: bool,

    /// Maximum number of packed tokens per varlen forward pass.
    ///
    /// `None` resolves to [`DEFAULT_MAX_TOKENS_PER_FORWARD`] (16 384).
    #[serde(default)]
    pub max_tokens_per_forward: Option<usize>,
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

fn default_local_rope_freq_base() -> f32 {
    DEFAULT_LOCAL_ROPE_FREQ_BASE
}

fn default_rms_eps() -> f64 {
    1e-6
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            intermediate_size: 0,
            max_position_embeddings: 512,
            layer_norm_eps: default_eps(),
            hidden_act: HiddenAct::default(),
            type_vocab_size: 0,
            arch_family: ArchFamily::default(),
            padding_token_id: 0,
            compute_dtype: default_compute_dtype(),
            rope_freq_base: default_rope_freq_base(),
            rope_freq_base_local: default_local_rope_freq_base(),
            causal: false,
            ffn_variant: FfnVariant::default(),
            norm_scheme: NormScheme::default(),
            token_type_embed_size: 0,
            num_kv_heads: 0,
            head_dim_explicit: None,
            rms_eps: default_rms_eps(),
            sliding_window: None,
            sliding_window_pattern: 0,
            position_embd_offset: 0,
            declared_pooling_type: None,
            alibi_max_bias: None,
            embed_scale: false,
            max_tokens_per_forward: None,
        }
    }
}

impl EncoderConfig {
    /// Compute the dimension of each attention head (`hidden_size / num_attention_heads`).
    ///
    /// This is the BERT/NomicBert formula. For architectures that store an
    /// explicit head dimension, use [`Self::resolved_head_dim`].
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Resolve the per-head dimension.
    ///
    /// Returns `head_dim_explicit` when set (from `<arch>.attention.key_length`).
    /// Falls back to `hidden_size / num_attention_heads` when `None`.
    pub fn resolved_head_dim(&self) -> usize {
        self.head_dim_explicit
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    /// Resolve the number of KV heads.
    ///
    /// Returns the stored `num_kv_heads` when non-zero (GQA).
    /// Falls back to `num_attention_heads` (MHA).
    pub fn resolved_num_kv_heads(&self) -> usize {
        if self.num_kv_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_kv_heads
        }
    }

    /// The position-table row a real token at 0-based rank `rank` maps to.
    ///
    /// XLM-RoBERTa numbers from `padding_token_id + 1`; every other family
    /// numbers from 0. [`Self::position_embd_offset`] then subtracts whatever
    /// the weight producer already chopped off the front of the table, so the
    /// same config drives both a HuggingFace table and a converted GGUF one.
    pub fn position_row(&self, rank: i64) -> i64 {
        let base = if self.arch_family == ArchFamily::XlmRoberta {
            self.padding_token_id + 1 + rank
        } else {
            rank
        };
        (base - self.position_embd_offset).max(0)
    }

    /// The position-table row padding occupies.
    ///
    /// Masked out of attention either way, but it must stay inside the table.
    pub fn padding_position_row(&self) -> i64 {
        if self.arch_family == ArchFamily::XlmRoberta {
            (self.padding_token_id - self.position_embd_offset).max(0)
        } else {
            0
        }
    }

    /// Whether this architecture interleaves local and global attention blocks.
    pub fn interleaves_attention(&self) -> bool {
        self.sliding_window.is_some() && self.sliding_window_pattern > 1
    }

    /// Whether block `layer_index` is a local (windowed, low-RoPE-base) block.
    ///
    /// Mirrors llama.cpp's `llama_hparams::set_swa_pattern` with
    /// `dense_first = false`: `is_swa[il] = il % pattern < pattern - 1`.
    /// Non-interleaved architectures have no local blocks.
    pub fn is_local_layer(&self, layer_index: usize) -> bool {
        self.interleaves_attention()
            && layer_index % self.sliding_window_pattern < self.sliding_window_pattern - 1
    }

    /// How block `layer_index` attends: RoPE base, window, and causality.
    ///
    /// The single source of truth. Builders take the RoPE base from here and
    /// the mask window from the same value, so the two cannot disagree.
    pub fn layer_attention(&self, layer_index: usize) -> LayerAttention {
        if self.is_local_layer(layer_index) {
            LayerAttention {
                rope_freq_base: self.rope_freq_base_local,
                window: self.sliding_window,
                causal: self.causal,
            }
        } else {
            LayerAttention {
                rope_freq_base: self.rope_freq_base,
                window: None,
                causal: self.causal,
            }
        }
    }

    /// Whether a packed (varlen) forward over sequences of at most `max_seqlen`
    /// tokens needs no span masking at all.
    ///
    /// The varlen attention kernel has no bounded-span support, so this must
    /// hold before that path may be used. A symmetric window is provably inert
    /// while the furthest pair in a sequence — `max_seqlen - 1` apart — is still
    /// within the half-width.
    pub fn varlen_span_is_unconstrained(&self, max_seqlen: usize) -> bool {
        if self.causal {
            return false;
        }
        // ALiBi is not a span constraint, but it reaches attention through the
        // same additive-bias slot the varlen kernel has no room for. Running
        // packed would drop the model's ONLY source of position information.
        if self.alibi_max_bias.is_some() {
            return false;
        }
        match self.layer_attention(0).max_distance() {
            Some(half) => max_seqlen <= 1 || max_seqlen - 1 <= half,
            None => true,
        }
    }

    /// The distinct RoPE bases this model needs, in build order.
    ///
    /// At most two: the global base, plus the local base when interleaving.
    /// Lets a builder allocate one cache per distinct base rather than one per
    /// block, and share them by `Arc`.
    pub fn distinct_rope_bases(&self) -> Vec<f32> {
        if self.interleaves_attention() && self.rope_freq_base_local != self.rope_freq_base {
            vec![self.rope_freq_base, self.rope_freq_base_local]
        } else {
            vec![self.rope_freq_base]
        }
    }
}
