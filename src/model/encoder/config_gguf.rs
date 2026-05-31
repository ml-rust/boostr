//! GGUF metadata → `EncoderConfig` parsing for BERT, NomicBert, and
//! Gemma-embedding backbones. Split from `config.rs` to keep each file focused.

use super::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct};
use numr::dtype::DType;

impl EncoderConfig {
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

        // Gemma-embedding has its own metadata namespace; dispatch first.
        if metadata.get_string("general.architecture") == Some("gemma-embedding") {
            return Self::from_gguf_metadata_gemma(metadata);
        }

        // NomicBert has its own metadata namespace; dispatch before the BERT path.
        if metadata.get_string("general.architecture") == Some("nomic-bert") {
            return Self::from_gguf_metadata_nomic(metadata);
        }

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
            rope_freq_base: 10000.0,
            causal: false,
            ffn_variant: FfnVariant::Standard,
            token_type_embed_size: 0,
            num_kv_heads: 0,
            head_dim_explicit: None,
            rms_eps: 1e-6,
            sliding_window: None,
            embed_scale: false,
            max_tokens_per_forward: None,
        })
    }

    /// Build an `EncoderConfig` from nomic-bert GGUF metadata keys.
    ///
    /// Reads `nomic-bert.*` metadata keys as specified by the GGUF ground truth:
    /// - `nomic-bert.embedding_length` → `hidden_size`
    /// - `nomic-bert.feed_forward_length` → `intermediate_size`
    /// - `nomic-bert.attention.head_count` → `num_attention_heads`
    /// - `nomic-bert.block_count` → `num_hidden_layers`
    /// - `nomic-bert.context_length` → `max_position_embeddings` (default 2048)
    /// - `nomic-bert.attention.layer_norm_epsilon` → `layer_norm_eps`
    /// - `nomic-bert.rope.freq_base` → `rope_freq_base` (default 10000.0)
    /// - `nomic-bert.attention.causal` → `causal` (expected false)
    /// - `nomic-bert.pooling_type` → validated as 1 (mean)
    fn from_gguf_metadata_nomic(
        metadata: &crate::format::GgufMetadata,
    ) -> crate::error::Result<Self> {
        use crate::error::Error;
        use crate::format::GgufValue;

        let hidden_size = metadata
            .get_u32("nomic-bert.embedding_length")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing nomic-bert.embedding_length".into(),
            })? as usize;

        let intermediate_size = metadata
            .get_u32("nomic-bert.feed_forward_length")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing nomic-bert.feed_forward_length".into(),
            })? as usize;

        let num_attention_heads = metadata
            .get_u32("nomic-bert.attention.head_count")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing nomic-bert.attention.head_count".into(),
            })? as usize;

        let num_hidden_layers =
            metadata
                .get_u32("nomic-bert.block_count")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing nomic-bert.block_count".into(),
                })? as usize;

        let max_position_embeddings = metadata
            .get_u32("nomic-bert.context_length")
            .unwrap_or(2048) as usize;

        let layer_norm_eps = metadata
            .get_f32("nomic-bert.attention.layer_norm_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-12);

        let rope_freq_base = metadata
            .get_f32("nomic-bert.rope.freq_base")
            .unwrap_or(10000.0);

        // `nomic-bert.attention.causal` is stored as a bool in GgufValue.
        let causal = metadata
            .get("nomic-bert.attention.causal")
            .and_then(|v| match v {
                GgufValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        // pooling_type = 1 means mean pooling; validate if present.
        if let Some(pt) = metadata.get_u32("nomic-bert.pooling_type") {
            if pt != 1 {
                return Err(Error::ModelError {
                    reason: format!(
                        "nomic-bert.pooling_type = {pt}; only mean pooling (1) is supported"
                    ),
                });
            }
        }

        let vocab_size = metadata
            .get_array("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .unwrap_or(30522);

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 2,
            arch_family: ArchFamily::NomicBert,
            padding_token_id: 0,
            compute_dtype: DType::F32,
            rope_freq_base,
            causal,
            ffn_variant: FfnVariant::GatedSilu,
            token_type_embed_size: 2,
            num_kv_heads: 0,
            head_dim_explicit: None,
            rms_eps: 1e-6,
            sliding_window: None,
            embed_scale: false,
            max_tokens_per_forward: None,
        })
    }

    /// Build an `EncoderConfig` from gemma-embedding GGUF metadata keys.
    ///
    /// Reads all `gemma-embedding.*` metadata keys. All fields are required
    /// except those with documented defaults.
    fn from_gguf_metadata_gemma(
        metadata: &crate::format::GgufMetadata,
    ) -> crate::error::Result<Self> {
        use crate::error::Error;

        let hidden_size = metadata
            .get_u32("gemma-embedding.embedding_length")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing gemma-embedding.embedding_length".into(),
            })? as usize;

        let intermediate_size = metadata
            .get_u32("gemma-embedding.feed_forward_length")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing gemma-embedding.feed_forward_length".into(),
            })? as usize;

        let num_attention_heads = metadata
            .get_u32("gemma-embedding.attention.head_count")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing gemma-embedding.attention.head_count".into(),
            })? as usize;

        let num_kv_heads = metadata
            .get_u32("gemma-embedding.attention.head_count_kv")
            .ok_or_else(|| Error::ModelError {
                reason: "GGUF missing gemma-embedding.attention.head_count_kv".into(),
            })? as usize;

        let head_dim_explicit = metadata
            .get_u32("gemma-embedding.attention.key_length")
            .map(|v| v as usize);

        let num_hidden_layers =
            metadata
                .get_u32("gemma-embedding.block_count")
                .ok_or_else(|| Error::ModelError {
                    reason: "GGUF missing gemma-embedding.block_count".into(),
                })? as usize;

        let max_position_embeddings = metadata
            .get_u32("gemma-embedding.context_length")
            .unwrap_or(8192) as usize;

        let rms_eps = metadata
            .get_f32("gemma-embedding.attention.layer_norm_rms_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-6);

        let sliding_window = metadata
            .get_u32("gemma-embedding.attention.sliding_window")
            .map(|v| v as usize);

        let rope_freq_base = metadata
            .get_f32("gemma-embedding.rope.freq_base")
            .unwrap_or(10000.0);

        if let Some(pt) = metadata.get_u32("gemma-embedding.pooling_type") {
            if pt != 1 {
                return Err(Error::ModelError {
                    reason: format!(
                        "gemma-embedding.pooling_type = {pt}; only mean pooling (1) is supported"
                    ),
                });
            }
        }

        let vocab_size = metadata
            .get_array("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .unwrap_or(256000); // Gemma default vocab size

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps: rms_eps,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
            arch_family: ArchFamily::GemmaEmbedding,
            padding_token_id: 0,
            compute_dtype: DType::F32,
            rope_freq_base,
            causal: false,
            ffn_variant: FfnVariant::GatedGelu,
            token_type_embed_size: 0,
            num_kv_heads,
            head_dim_explicit,
            rms_eps,
            sliding_window,
            embed_scale: true,
            max_tokens_per_forward: None,
        })
    }
}
