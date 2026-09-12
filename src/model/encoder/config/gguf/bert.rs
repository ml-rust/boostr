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

#[cfg(test)]
mod tests {
    use super::super::dispatch::tests::meta;
    use super::*;
    use crate::format::GgufValue;

    /// `bge-m3` ships a `bert`-namespace GGUF that declares CLS pooling. The
    /// namespace serves both mean- and CLS-pooled encoders, so the architecture
    /// default is not the answer and the file has to be read.
    #[test]
    fn bert_namespace_carries_the_declared_pooling_type() {
        let cls = meta(&[
            ("general.architecture", GgufValue::String("bert".into())),
            ("bert.embedding_length", GgufValue::Uint32(1024)),
            ("bert.feed_forward_length", GgufValue::Uint32(4096)),
            ("bert.attention.head_count", GgufValue::Uint32(16)),
            ("bert.block_count", GgufValue::Uint32(24)),
            ("bert.context_length", GgufValue::Uint32(8192)),
            ("bert.pooling_type", GgufValue::Uint32(2)),
            ("tokenizer.ggml.model", GgufValue::String("t5".into())),
        ]);
        let config = EncoderConfig::from_gguf_metadata(&cls).unwrap();
        assert_eq!(config.declared_pooling_type, Some(2));

        let mean = meta(&[
            ("general.architecture", GgufValue::String("bert".into())),
            ("bert.embedding_length", GgufValue::Uint32(384)),
            ("bert.feed_forward_length", GgufValue::Uint32(1536)),
            ("bert.attention.head_count", GgufValue::Uint32(12)),
            ("bert.block_count", GgufValue::Uint32(6)),
            ("bert.pooling_type", GgufValue::Uint32(1)),
        ]);
        let config = EncoderConfig::from_gguf_metadata(&mean).unwrap();
        assert_eq!(config.declared_pooling_type, Some(1));
    }

    /// A converted XLM-RoBERTa GGUF has its dead leading position rows chopped off,
    /// so the first real token must read row 0 — not row `pad_id + 1`. A config
    /// built from HuggingFace weights keeps the offset, because that table is
    /// intact.
    #[test]
    fn xlm_roberta_position_rows_are_rebased_for_gguf_only() {
        let gguf = meta(&[
            ("general.architecture", GgufValue::String("bert".into())),
            ("bert.embedding_length", GgufValue::Uint32(1024)),
            ("bert.feed_forward_length", GgufValue::Uint32(4096)),
            ("bert.attention.head_count", GgufValue::Uint32(16)),
            ("bert.block_count", GgufValue::Uint32(24)),
            ("tokenizer.ggml.model", GgufValue::String("t5".into())),
        ]);
        let config = EncoderConfig::from_gguf_metadata(&gguf).unwrap();
        assert_eq!(config.arch_family, ArchFamily::XlmRoberta);
        assert_eq!(config.padding_token_id, 1);
        assert_eq!(config.position_embd_offset, 2);
        assert_eq!(config.position_row(0), 0);
        assert_eq!(config.position_row(5), 5);
        assert_eq!(config.padding_position_row(), 0);

        // Same family, weights straight from HuggingFace: nothing was chopped.
        let hf = EncoderConfig {
            arch_family: ArchFamily::XlmRoberta,
            padding_token_id: 1,
            position_embd_offset: 0,
            ..Default::default()
        };
        assert_eq!(hf.position_row(0), 2);
        assert_eq!(hf.position_row(5), 7);
        assert_eq!(hf.padding_position_row(), 1);

        // A plain BERT config is 0-based and unaffected by either knob.
        let bert = EncoderConfig::default();
        assert_eq!(bert.position_row(0), 0);
        assert_eq!(bert.position_row(9), 9);
    }
}
