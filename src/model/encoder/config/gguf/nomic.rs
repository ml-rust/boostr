//! `nomic-bert.*` GGUF namespace.

use super::dispatch::{require_pooling_type, required_u32, vocab_size};
use crate::error::Result;
use crate::format::{GgufMetadata, GgufValue};
use crate::model::encoder::config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct, NormScheme};

impl EncoderConfig {
    /// Build from the `nomic-bert.*` GGUF namespace.
    ///
    /// NomicBert replaces learned position embeddings with RoPE, uses a SwiGLU
    /// FFN and a fused QKV projection, and pools by mean.
    pub(super) fn from_gguf_metadata_nomic(metadata: &GgufMetadata) -> Result<Self> {
        let hidden_size = required_u32(metadata, "nomic-bert.embedding_length")?;
        let intermediate_size = required_u32(metadata, "nomic-bert.feed_forward_length")?;
        let num_attention_heads = required_u32(metadata, "nomic-bert.attention.head_count")?;
        let num_hidden_layers = required_u32(metadata, "nomic-bert.block_count")?;

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

        let causal = metadata
            .get("nomic-bert.attention.causal")
            .and_then(|v| match v {
                GgufValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        require_pooling_type(metadata, "nomic-bert.pooling_type", &[1])?;

        Ok(Self {
            vocab_size: vocab_size(metadata, 30522),
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 2,
            arch_family: ArchFamily::NomicBert,
            rope_freq_base,
            causal,
            ffn_variant: FfnVariant::GatedSilu,
            norm_scheme: NormScheme::PostNorm,
            token_type_embed_size: 2,
            ..Default::default()
        })
    }
}
