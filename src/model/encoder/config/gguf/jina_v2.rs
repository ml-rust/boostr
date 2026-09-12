//! `jina-bert-v2.*` GGUF namespace — jina-embeddings-v2 backbones.

use super::dispatch::{require_pooling_type, required_u32, vocab_size};
use crate::error::Result;
use crate::format::{GgufMetadata, GgufValue};
use crate::model::encoder::config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct, NormScheme};

/// Maximum ALiBi bias for jina-bert-v2.
///
/// Hard-coded rather than read from the file because llama.cpp hard-codes it
/// too (`hparams.f_max_alibi_bias = 8.0f` in the `LLM_ARCH_JINA_BERT_V2` arm)
/// and no converter writes a corresponding GGUF key. A default of 0 would
/// silently flatten every slope to 1.0.
const JINA_V2_MAX_ALIBI_BIAS: f32 = 8.0;

impl EncoderConfig {
    /// Build from the `jina-bert-v2.*` GGUF namespace.
    ///
    /// jina-embeddings-v2 is the only encoder here whose positions come from
    /// ALiBi: the file carries neither a `position_embd` table nor a
    /// `rope.freq_base` key, so position information exists ONLY as the
    /// per-head distance penalty added to the attention scores. Loading it
    /// without that penalty produces a bag-of-words encoder that still returns
    /// well-shaped, plausibly-scaled vectors.
    ///
    /// The rest: separate biased Q/K/V, LayerNorm QK-norm over the whole hidden
    /// vector, a second post-attention norm (`attn_norm_2`), and a GeGLU FFN
    /// whose bias sits on `ffn_down` alone.
    pub(super) fn from_gguf_metadata_jina_v2(metadata: &GgufMetadata) -> Result<Self> {
        let hidden_size = required_u32(metadata, "jina-bert-v2.embedding_length")?;
        let intermediate_size = required_u32(metadata, "jina-bert-v2.feed_forward_length")?;
        let num_attention_heads = required_u32(metadata, "jina-bert-v2.attention.head_count")?;
        let num_hidden_layers = required_u32(metadata, "jina-bert-v2.block_count")?;

        let num_kv_heads = metadata
            .get_u32("jina-bert-v2.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let max_position_embeddings = metadata
            .get_u32("jina-bert-v2.context_length")
            .unwrap_or(8192) as usize;

        let layer_norm_eps = metadata
            .get_f32("jina-bert-v2.attention.layer_norm_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-12);

        let causal = metadata
            .get("jina-bert-v2.attention.causal")
            .and_then(|v| match v {
                GgufValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        require_pooling_type(metadata, "jina-bert-v2.pooling_type", &[1])?;

        Ok(Self {
            vocab_size: vocab_size(metadata, 61056),
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 2,
            token_type_embed_size: 2,
            arch_family: ArchFamily::JinaBertV2,
            causal,
            num_kv_heads,
            alibi_max_bias: Some(JINA_V2_MAX_ALIBI_BIAS),
            ffn_variant: FfnVariant::GatedGelu,
            norm_scheme: NormScheme::PostNorm,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::tests::meta;
    use super::*;

    /// The metadata actually present in `jina-embeddings-v2-base-code-Q8_0.gguf`.
    /// Note the absent `rope.freq_base`: this file has no rotary key at all.
    fn jina_v2_metadata() -> GgufMetadata {
        meta(&[
            (
                "general.architecture",
                GgufValue::String("jina-bert-v2".into()),
            ),
            ("jina-bert-v2.embedding_length", GgufValue::Uint32(768)),
            ("jina-bert-v2.feed_forward_length", GgufValue::Uint32(3072)),
            ("jina-bert-v2.attention.head_count", GgufValue::Uint32(12)),
            ("jina-bert-v2.block_count", GgufValue::Uint32(12)),
            ("jina-bert-v2.context_length", GgufValue::Uint32(8192)),
            (
                "jina-bert-v2.attention.layer_norm_epsilon",
                GgufValue::Float32(1e-12),
            ),
            ("jina-bert-v2.attention.causal", GgufValue::Bool(false)),
            ("jina-bert-v2.pooling_type", GgufValue::Uint32(1)),
        ])
    }

    /// jina-bert-v2 carries neither a rotary key nor a position table, so ALiBi is
    /// its only source of position. A config that silently left `alibi_max_bias`
    /// unset would load and run as a bag-of-words encoder.
    #[test]
    fn jina_v2_config_enables_alibi_and_no_rope() {
        let config = EncoderConfig::from_gguf_metadata(&jina_v2_metadata()).unwrap();

        assert_eq!(config.arch_family, ArchFamily::JinaBertV2);
        assert!(!config.arch_family.uses_rope());
        assert!(config.arch_family.uses_alibi());
        assert!(!config.arch_family.uses_learned_positions());
        assert_eq!(config.alibi_max_bias, Some(8.0));
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.ffn_variant, FfnVariant::GatedGelu);
        assert_eq!(config.norm_scheme, NormScheme::PostNorm);
        assert!(!config.causal);
        assert_eq!(config.sliding_window, None);
    }

    /// The packed path cannot apply an additive per-head bias, so an ALiBi model
    /// must be refused there rather than silently returning position-free vectors.
    #[test]
    fn alibi_forces_the_padded_path() {
        let config = EncoderConfig::from_gguf_metadata(&jina_v2_metadata()).unwrap();
        assert!(!config.varlen_span_is_unconstrained(8));
        assert!(!config.varlen_span_is_unconstrained(1));
    }
}
