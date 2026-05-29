//! Encoder model configuration (BERT-style transformer encoders).

use numr::dtype::DType;
use serde::{Deserialize, Serialize};

/// Architecture family for position-id generation and embedding behaviour.
///
/// BERT uses simple 0-based position ids.  XLM-RoBERTa (used by e.g.
/// bge-reranker-v2-m3) reserves position `pad_token_id` for padding and
/// numbers real tokens starting from `pad_token_id + 1`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArchFamily {
    /// Standard BERT: position_ids = [0, 1, ..., S-1].
    #[default]
    Bert,
    /// XLM-RoBERTa: position_ids computed as cumsum(input_ids != pad_id) + pad_id,
    /// with padding positions assigned position_id = pad_id.
    XlmRoberta,
}

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

impl EncoderConfig {
    /// Compute the dimension of each attention head (`hidden_size / num_attention_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Build an `EncoderConfig` from GGUF metadata keys.
    ///
    /// Reads standard BERT GGUF keys:
    /// - `bert.embedding_length` → `hidden_size`
    /// - `bert.feed_forward_length` → `intermediate_size`
    /// - `bert.attention.head_count` → `num_attention_heads`
    /// - `bert.block_count` → `num_hidden_layers`
    /// - `bert.context_length` → `max_position_embeddings`
    ///
    /// Vocab size is inferred from the `tokenizer.ggml.tokens` array length.
    ///
    /// Architecture family is inferred from the tokenizer model:
    /// `tokenizer.ggml.model == "t5"` (SentencePiece / unigram) indicates
    /// an XLM-RoBERTa backbone and sets `arch_family = XlmRoberta` with
    /// `padding_token_id = 1`.  BERT-family models use BPE/WordPiece and
    /// get `arch_family = Bert` with `padding_token_id = 0`.
    pub fn from_gguf_metadata(
        metadata: &crate::format::GgufMetadata,
    ) -> crate::error::Result<Self> {
        use crate::error::Error;

        let hidden_size =
            metadata
                .get_u32("bert.embedding_length")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing bert.embedding_length".into(),
                })? as usize;

        let intermediate_size = metadata
            .get_u32("bert.feed_forward_length")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing bert.feed_forward_length".into(),
            })? as usize;

        let num_attention_heads =
            metadata
                .get_u32("bert.attention.head_count")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing bert.attention.head_count".into(),
                })? as usize;

        let num_hidden_layers =
            metadata
                .get_u32("bert.block_count")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing bert.block_count".into(),
                })? as usize;

        let max_position_embeddings =
            metadata.get_u32("bert.context_length").unwrap_or(512) as usize;

        let vocab_size = metadata
            .get_array("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .unwrap_or(30522); // BERT default

        // Detect XLM-RoBERTa: GGUF tokenizer model "t5" means SentencePiece/unigram,
        // which is the tokenizer used by all XLM-RoBERTa variants.  BERT/RoBERTa-base
        // models use "bert" or "bpe" here.
        let tokenizer_model = metadata
            .get_string("tokenizer.ggml.model")
            .unwrap_or("bert");
        let (arch_family, padding_token_id) = if tokenizer_model == "t5" {
            // XLM-RoBERTa: <pad> is always at position 1 in the SentencePiece vocabulary.
            (ArchFamily::XlmRoberta, 1i64)
        } else {
            (ArchFamily::Bert, 0i64)
        };

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
            arch_family,
            padding_token_id,
            compute_dtype: DType::F32,
        })
    }
}
