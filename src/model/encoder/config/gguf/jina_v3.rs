//! `jina-bert-v3.*` GGUF namespace — jina-embeddings-v3 backbones.

use super::dispatch::{require_pooling_type, required_u32, vocab_size};
use crate::error::Result;
use crate::format::{GgufMetadata, GgufValue};
use crate::model::encoder::config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct, NormScheme};

impl EncoderConfig {
    /// Build from the `jina-bert-v3.*` GGUF namespace.
    ///
    /// jina-embeddings-v3 is an XLM-RoBERTa-shaped encoder that swaps the
    /// learned position table for RoPE. The HuggingFace repo still reports
    /// `XLMRobertaModel`, so the name is not a safe guide: loaded through the
    /// BERT/XLM-R path it would look for a `position_embd` tensor the file does
    /// not contain, and its rotary positions would never be applied.
    ///
    /// Otherwise it is plain post-norm BERT: fused biased QKV, a standard
    /// (non-gated) GELU FFN with biases, LayerNorm everywhere, mean pooling.
    ///
    /// The published task-specific numbers come from five LoRA adapters
    /// (`retrieval.query`, `retrieval.passage`, …). The GGUF carries the
    /// backbone only — there are no LoRA tensors in the file — so absolute
    /// quality sits below the paper's benchmarks.
    pub(super) fn from_gguf_metadata_jina_v3(metadata: &GgufMetadata) -> Result<Self> {
        let hidden_size = required_u32(metadata, "jina-bert-v3.embedding_length")?;
        let intermediate_size = required_u32(metadata, "jina-bert-v3.feed_forward_length")?;
        let num_attention_heads = required_u32(metadata, "jina-bert-v3.attention.head_count")?;
        let num_hidden_layers = required_u32(metadata, "jina-bert-v3.block_count")?;

        let max_position_embeddings = metadata
            .get_u32("jina-bert-v3.context_length")
            .unwrap_or(8192) as usize;

        let layer_norm_eps = metadata
            .get_f32("jina-bert-v3.attention.layer_norm_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-12);

        // 20 000, not the 10 000 every other RoPE encoder here uses. Reading it
        // is not optional: the wrong base rotates every key by the wrong angle
        // and degrades retrieval without changing a single shape.
        let rope_freq_base = metadata
            .get_f32("jina-bert-v3.rope.freq_base")
            .unwrap_or(10000.0);

        let causal = metadata
            .get("jina-bert-v3.attention.causal")
            .and_then(|v| match v {
                GgufValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        require_pooling_type(metadata, "jina-bert-v3.pooling_type", &[1])?;

        Ok(Self {
            vocab_size: vocab_size(metadata, 250002),
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps,
            hidden_act: HiddenAct::Gelu,
            // `token_types.weight` is one row wide in this file, so there is a
            // single segment type and row 0 is the whole tensor.
            type_vocab_size: 1,
            token_type_embed_size: 1,
            arch_family: ArchFamily::JinaBertV3,
            rope_freq_base,
            causal,
            ffn_variant: FfnVariant::Standard,
            norm_scheme: NormScheme::PostNorm,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::tests::meta;
    use super::*;

    /// The metadata actually present in `jina-embeddings-v3-Q4_0.gguf`.
    fn jina_v3_metadata() -> GgufMetadata {
        meta(&[
            (
                "general.architecture",
                GgufValue::String("jina-bert-v3".into()),
            ),
            ("jina-bert-v3.embedding_length", GgufValue::Uint32(1024)),
            ("jina-bert-v3.feed_forward_length", GgufValue::Uint32(4096)),
            ("jina-bert-v3.attention.head_count", GgufValue::Uint32(16)),
            ("jina-bert-v3.block_count", GgufValue::Uint32(24)),
            ("jina-bert-v3.context_length", GgufValue::Uint32(8192)),
            (
                "jina-bert-v3.attention.layer_norm_epsilon",
                GgufValue::Float32(1e-5),
            ),
            ("jina-bert-v3.attention.causal", GgufValue::Bool(false)),
            ("jina-bert-v3.pooling_type", GgufValue::Uint32(1)),
            ("jina-bert-v3.rope.freq_base", GgufValue::Float32(20000.0)),
        ])
    }

    /// jina-bert-v3 must route to its own family, not to the BERT fallback: it
    /// reports `XLMRobertaModel` in its HuggingFace config but has no `position_embd` tensor, and
    /// its rotary base is 20 000 rather than the usual 10 000.
    #[test]
    fn jina_v3_config_uses_rope_at_its_own_base() {
        let config = EncoderConfig::from_gguf_metadata(&jina_v3_metadata()).unwrap();

        assert_eq!(config.arch_family, ArchFamily::JinaBertV3);
        assert!(config.arch_family.uses_rope());
        assert!(!config.arch_family.uses_learned_positions());
        assert_eq!(config.rope_freq_base, 20000.0);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.ffn_variant, FfnVariant::Standard);
        assert_eq!(config.norm_scheme, NormScheme::PostNorm);
        assert!(!config.causal);
        assert!(config.alibi_max_bias.is_none());
        assert!((config.layer_norm_eps - 1e-5).abs() < 1e-12);
    }
}
