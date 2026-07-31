//! `bert.*` GGUF namespace — also serves XLM-RoBERTa backbones.

use super::dispatch::{required_u32, vocab_size};
use crate::error::Result;
use crate::format::GgufMetadata;
use crate::model::encoder::config::{ArchFamily, EncoderConfig, HiddenAct, NormScheme};

impl EncoderConfig {
    /// Build from the standard BERT GGUF namespace.
    ///
    /// Architecture family is inferred from the tokenizer model:
    /// `tokenizer.ggml.model == "t5"` (SentencePiece / unigram) indicates an
    /// XLM-RoBERTa backbone, which reserves position `pad_token_id` for padding.
    /// BERT-family models use BPE/WordPiece.
    pub(super) fn from_gguf_metadata_bert(metadata: &GgufMetadata) -> Result<Self> {
        let hidden_size = required_u32(metadata, "bert.embedding_length")?;
        let intermediate_size = required_u32(metadata, "bert.feed_forward_length")?;
        let num_attention_heads = required_u32(metadata, "bert.attention.head_count")?;
        let num_hidden_layers = required_u32(metadata, "bert.block_count")?;
        let max_position_embeddings =
            metadata.get_u32("bert.context_length").unwrap_or(512) as usize;

        // XLM-RoBERTa: <pad> is always at position 1 in the SentencePiece vocabulary.
        let tokenizer_model = metadata
            .get_string("tokenizer.ggml.model")
            .unwrap_or("bert");
        let (arch_family, padding_token_id) = if tokenizer_model == "t5" {
            (ArchFamily::XlmRoberta, 1i64)
        } else {
            (ArchFamily::Bert, 0i64)
        };

        Ok(Self {
            vocab_size: vocab_size(metadata, 30522),
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            arch_family,
            padding_token_id,
            // llama.cpp's converter chops the dead leading rows off an
            // XLM-RoBERTa position table, so a GGUF one is already re-based.
            position_embd_offset: if arch_family == ArchFamily::XlmRoberta {
                padding_token_id + 1
            } else {
                0
            },
            norm_scheme: NormScheme::PostNorm,
            // Unlike the dedicated namespaces, this one does NOT constrain the
            // pooling type to a single value: the `bert` namespace serves both
            // mean-pooled sentence encoders and CLS-pooled ones (bge-m3
            // declares 2). Carry whatever the file says and let `Pooling`
            // resolve it.
            declared_pooling_type: metadata.get_u32("bert.pooling_type"),
            ..Default::default()
        })
    }
}
