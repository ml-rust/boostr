//! `gemma-embedding.*` GGUF namespace.

use super::dispatch::{require_pooling_type, required_u32, vocab_size};
use crate::error::{Error, Result};
use crate::format::GgufMetadata;
use crate::model::encoder::config::encoder_config::{
    DEFAULT_LOCAL_ROPE_FREQ_BASE, DEFAULT_SLIDING_WINDOW_PATTERN,
};
use crate::model::encoder::config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct, NormScheme};

/// Metadata keys for the optional sentence-transformers Dense bottleneck
/// modules. Present only when the file was converted with
/// `--sentence-transformers-dense-modules`.
const DENSE_MODULE_KEYS: &[&str] = &[
    "gemma-embedding.dense_2_feat_in",
    "gemma-embedding.dense_2_feat_out",
    "gemma-embedding.dense_3_feat_in",
    "gemma-embedding.dense_3_feat_out",
];

impl EncoderConfig {
    /// Build from the `gemma-embedding.*` GGUF namespace.
    ///
    /// EmbeddingGemma is Gemma3-based and interleaves two attention types on a
    /// fixed block period. Neither the local RoPE base nor the block period is
    /// usually present in the file; both fall back to the architecture defaults
    /// that llama.cpp applies (10 000 and 6 respectively).
    pub(super) fn from_gguf_metadata_gemma(metadata: &GgufMetadata) -> Result<Self> {
        let hidden_size = required_u32(metadata, "gemma-embedding.embedding_length")?;
        let intermediate_size = required_u32(metadata, "gemma-embedding.feed_forward_length")?;
        let num_attention_heads = required_u32(metadata, "gemma-embedding.attention.head_count")?;
        let num_kv_heads = required_u32(metadata, "gemma-embedding.attention.head_count_kv")?;
        let num_hidden_layers = required_u32(metadata, "gemma-embedding.block_count")?;

        let head_dim_explicit = metadata
            .get_u32("gemma-embedding.attention.key_length")
            .map(|v| v as usize);

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

        // Absent in every published EmbeddingGemma file. llama.cpp leaves
        // `rope_freq_base_train_swa` at its 10 000 initialiser rather than
        // backfilling it from the global base, so local blocks genuinely rotate
        // at a different base from global ones.
        let rope_freq_base_local = metadata
            .get_f32("gemma-embedding.rope.freq_base_swa")
            .unwrap_or(DEFAULT_LOCAL_ROPE_FREQ_BASE);

        // Also absent in published files; 6 is the architecture default.
        let sliding_window_pattern = metadata
            .get_u32("gemma-embedding.attention.sliding_window_pattern")
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN);

        reject_dense_modules(metadata)?;
        require_pooling_type(metadata, "gemma-embedding.pooling_type", &[1])?;

        Ok(Self {
            vocab_size: vocab_size(metadata, 256000),
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            layer_norm_eps: rms_eps,
            hidden_act: HiddenAct::Gelu,
            arch_family: ArchFamily::GemmaEmbedding,
            rope_freq_base,
            rope_freq_base_local,
            ffn_variant: FfnVariant::GatedGelu,
            norm_scheme: NormScheme::Sandwich,
            num_kv_heads,
            head_dim_explicit,
            rms_eps,
            sliding_window,
            sliding_window_pattern,
            embed_scale: true,
            ..Default::default()
        })
    }
}

/// Fail loudly on a file carrying sentence-transformers Dense bottleneck
/// modules.
///
/// Those files expect two extra projections after pooling. Loading one and
/// ignoring them would return plausible-looking vectors from the wrong
/// projection space — the exact silent-wrong-output failure this module exists
/// to avoid — so refuse instead.
fn reject_dense_modules(metadata: &GgufMetadata) -> Result<()> {
    let present: Vec<&str> = DENSE_MODULE_KEYS
        .iter()
        .copied()
        .filter(|k| metadata.get_u32(k).is_some_and(|v| v != 0))
        .collect();

    if present.is_empty() {
        return Ok(());
    }

    Err(Error::ModelError {
        reason: format!(
            "this EmbeddingGemma file carries sentence-transformers Dense modules \
             ({}), which are applied after pooling and are not implemented here. \
             Loading it would silently return vectors from the wrong projection \
             space. Use a GGUF converted without --sentence-transformers-dense-modules.",
            present.join(", ")
        ),
    })
}
