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

#[cfg(test)]
mod tests {
    use super::super::dispatch::tests::meta;
    use super::*;
    use crate::format::GgufValue;

    /// The metadata actually present in a published EmbeddingGemma GGUF. Notably it
    /// carries neither `rope.freq_base_swa` nor `attention.sliding_window_pattern`,
    /// so both must come from architecture defaults.
    fn embedding_gemma_metadata() -> GgufMetadata {
        meta(&[
            (
                "general.architecture",
                GgufValue::String("gemma-embedding".into()),
            ),
            ("gemma-embedding.embedding_length", GgufValue::Uint32(768)),
            (
                "gemma-embedding.feed_forward_length",
                GgufValue::Uint32(1152),
            ),
            ("gemma-embedding.attention.head_count", GgufValue::Uint32(3)),
            (
                "gemma-embedding.attention.head_count_kv",
                GgufValue::Uint32(1),
            ),
            (
                "gemma-embedding.attention.key_length",
                GgufValue::Uint32(256),
            ),
            (
                "gemma-embedding.attention.value_length",
                GgufValue::Uint32(256),
            ),
            ("gemma-embedding.block_count", GgufValue::Uint32(24)),
            ("gemma-embedding.context_length", GgufValue::Uint32(2048)),
            (
                "gemma-embedding.attention.sliding_window",
                GgufValue::Uint32(512),
            ),
            (
                "gemma-embedding.rope.freq_base",
                GgufValue::Float32(1_000_000.0),
            ),
            ("gemma-embedding.pooling_type", GgufValue::Uint32(1)),
        ])
    }

    #[test]
    fn gemma_local_rope_base_defaults_when_the_file_omits_it() {
        let cfg = EncoderConfig::from_gguf_metadata(&embedding_gemma_metadata()).unwrap();

        assert_eq!(
            cfg.rope_freq_base, 1_000_000.0,
            "global base comes from the file"
        );
        assert_eq!(
            cfg.rope_freq_base_local, 10_000.0,
            "the local base must fall back to the architecture default, NOT to the \
             global base — llama.cpp leaves rope_freq_base_train_swa at its 10 000 \
             initialiser when the key is absent"
        );
    }

    #[test]
    fn gemma_local_rope_base_is_read_when_present() {
        let mut m = embedding_gemma_metadata();
        m.kv.insert(
            "gemma-embedding.rope.freq_base_swa".into(),
            GgufValue::Float32(50_000.0),
        );
        let cfg = EncoderConfig::from_gguf_metadata(&m).unwrap();
        assert_eq!(cfg.rope_freq_base_local, 50_000.0);
    }

    #[test]
    fn gemma_sliding_window_pattern_defaults_to_six() {
        let cfg = EncoderConfig::from_gguf_metadata(&embedding_gemma_metadata()).unwrap();
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_eq!(cfg.sliding_window, Some(512));
    }

    #[test]
    fn gemma_sliding_window_pattern_is_read_when_present() {
        let mut m = embedding_gemma_metadata();
        m.kv.insert(
            "gemma-embedding.attention.sliding_window_pattern".into(),
            GgufValue::Uint32(4),
        );
        let cfg = EncoderConfig::from_gguf_metadata(&m).unwrap();
        assert_eq!(cfg.sliding_window_pattern, 4);
    }

    #[test]
    fn gemma_marks_twenty_of_twentyfour_blocks_local() {
        let cfg = EncoderConfig::from_gguf_metadata(&embedding_gemma_metadata()).unwrap();

        let global: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&il| !cfg.is_local_layer(il))
            .collect();

        assert_eq!(
            global,
            vec![5, 11, 17, 23],
            "with period 6, blocks at index % 6 == 5 are global"
        );
        let local_count = cfg.num_hidden_layers - global.len();
        assert_eq!(local_count, 20);
    }

    #[test]
    fn gemma_local_blocks_get_the_local_base_and_the_window() {
        let cfg = EncoderConfig::from_gguf_metadata(&embedding_gemma_metadata()).unwrap();

        let local = cfg.layer_attention(0);
        assert_eq!(local.rope_freq_base, 10_000.0);
        assert_eq!(local.window, Some(512));
        assert_eq!(
            local.max_distance(),
            Some(256),
            "the window is symmetric, so a 512-position window permits a distance of 256"
        );

        let global = cfg.layer_attention(5);
        assert_eq!(global.rope_freq_base, 1_000_000.0);
        assert_eq!(global.window, None);
        assert_eq!(global.max_distance(), None);
    }

    #[test]
    fn gemma_needs_exactly_two_rope_caches() {
        let cfg = EncoderConfig::from_gguf_metadata(&embedding_gemma_metadata()).unwrap();
        assert_eq!(cfg.distinct_rope_bases(), vec![1_000_000.0, 10_000.0]);
    }

    #[test]
    fn gemma_rejects_dense_module_files() {
        let mut m = embedding_gemma_metadata();
        m.kv.insert(
            "gemma-embedding.dense_2_feat_in".into(),
            GgufValue::Uint32(768),
        );
        m.kv.insert(
            "gemma-embedding.dense_2_feat_out".into(),
            GgufValue::Uint32(3072),
        );

        let err = EncoderConfig::from_gguf_metadata(&m)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Dense modules"),
            "a Dense-module file must be refused, not loaded with the projections \
             silently dropped; got: {err}"
        );
    }
}
