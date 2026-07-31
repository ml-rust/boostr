//! `qwen3.*` GGUF namespace — Qwen3-Embedding backbones.

use super::dispatch::{require_pooling_type, required_u32, vocab_size};
use crate::error::Result;
use crate::format::GgufMetadata;
use crate::model::encoder::config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct, NormScheme};

impl EncoderConfig {
    /// Build from the `qwen3.*` GGUF namespace.
    ///
    /// Qwen3-Embedding is a decoder backbone used as an embedder: attention is
    /// causal and the sentence vector is the hidden state of the last real
    /// token, not a mean. It uses pre-norm RMSNorm with plain residuals, GQA,
    /// QK-norm and a SwiGLU FFN.
    ///
    /// `attention.key_length` must be read rather than derived: for the 0.6B
    /// model it is 128 while `embedding_length / head_count` is 64, so the Q/K/V
    /// projections are wider than the residual stream.
    pub(super) fn from_gguf_metadata_qwen3(metadata: &GgufMetadata) -> Result<Self> {
        let hidden_size = required_u32(metadata, "qwen3.embedding_length")?;
        let intermediate_size = required_u32(metadata, "qwen3.feed_forward_length")?;
        let num_attention_heads = required_u32(metadata, "qwen3.attention.head_count")?;
        let num_hidden_layers = required_u32(metadata, "qwen3.block_count")?;

        let num_kv_heads = metadata
            .get_u32("qwen3.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let head_dim_explicit = metadata
            .get_u32("qwen3.attention.key_length")
            .map(|v| v as usize);

        let max_position_embeddings =
            metadata.get_u32("qwen3.context_length").unwrap_or(32768) as usize;

        let rms_eps = metadata
            .get_f32("qwen3.attention.layer_norm_rms_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-6);

        let rope_freq_base = metadata.get_f32("qwen3.rope.freq_base").unwrap_or(10000.0);

        // 3 = last-token pooling, which is what Qwen3-Embedding is trained for.
        // Mean pooling over a causal model would blend prefix states that never
        // saw the full input, so only 3 is accepted.
        require_pooling_type(metadata, "qwen3.pooling_type", &[3])?;

        Ok(Self {
            vocab_size: vocab_size(metadata, 151936),
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps: rms_eps,
            hidden_act: HiddenAct::Gelu,
            arch_family: ArchFamily::Qwen3,
            rope_freq_base,
            causal: true,
            ffn_variant: FfnVariant::GatedSilu,
            norm_scheme: NormScheme::PreNorm,
            num_kv_heads,
            head_dim_explicit,
            rms_eps,
            ..Default::default()
        })
    }
}
