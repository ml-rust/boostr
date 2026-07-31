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
