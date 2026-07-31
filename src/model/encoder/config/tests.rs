//! Config parsing and per-layer attention derivation.

use super::*;
use crate::format::{GgufMetadata, GgufValue};
use std::collections::HashMap;

/// Build metadata from `(key, value)` pairs.
fn meta(pairs: &[(&str, GgufValue)]) -> GgufMetadata {
    let mut kv = HashMap::new();
    for (k, v) in pairs {
        kv.insert((*k).to_string(), v.clone());
    }
    GgufMetadata { kv }
}

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

fn qwen3_metadata() -> GgufMetadata {
    meta(&[
        ("general.architecture", GgufValue::String("qwen3".into())),
        ("qwen3.embedding_length", GgufValue::Uint32(1024)),
        ("qwen3.feed_forward_length", GgufValue::Uint32(3072)),
        ("qwen3.attention.head_count", GgufValue::Uint32(16)),
        ("qwen3.attention.head_count_kv", GgufValue::Uint32(8)),
        ("qwen3.attention.key_length", GgufValue::Uint32(128)),
        ("qwen3.attention.value_length", GgufValue::Uint32(128)),
        ("qwen3.block_count", GgufValue::Uint32(28)),
        ("qwen3.context_length", GgufValue::Uint32(32768)),
        ("qwen3.rope.freq_base", GgufValue::Float32(1_000_000.0)),
        ("qwen3.pooling_type", GgufValue::Uint32(3)),
    ])
}

// ── Gemma: interleave parameters ─────────────────────────────────────────────

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

// ── Non-interleaved architectures ────────────────────────────────────────────

#[test]
fn architectures_without_a_sliding_window_never_report_local_blocks() {
    let cfg = EncoderConfig {
        num_hidden_layers: 12,
        sliding_window: None,
        sliding_window_pattern: 6,
        ..Default::default()
    };

    assert!(!cfg.interleaves_attention());
    for il in 0..cfg.num_hidden_layers {
        assert!(!cfg.is_local_layer(il), "block {il} must be global");
        assert_eq!(cfg.layer_attention(il).window, None);
        assert_eq!(cfg.layer_attention(il).rope_freq_base, cfg.rope_freq_base);
    }
    assert_eq!(cfg.distinct_rope_bases().len(), 1);
}

#[test]
fn a_zero_or_unit_pattern_disables_interleaving() {
    for pattern in [0usize, 1] {
        let cfg = EncoderConfig {
            num_hidden_layers: 12,
            sliding_window: Some(512),
            sliding_window_pattern: pattern,
            ..Default::default()
        };
        assert!(
            !cfg.interleaves_attention(),
            "pattern {pattern} must not interleave"
        );
        assert!((0..12).all(|il| !cfg.is_local_layer(il)));
    }
}

// ── Attention predicate ──────────────────────────────────────────────────────

#[test]
fn a_windowed_block_attends_symmetrically_in_both_directions() {
    let a = LayerAttention {
        rope_freq_base: 10_000.0,
        window: Some(8),
        causal: false,
    };

    assert!(a.attends(10, 10));
    assert!(a.attends(10, 14), "forward within the half-window");
    assert!(a.attends(10, 6), "backward within the half-window");
    assert!(!a.attends(10, 15), "one past the half-window");
    assert!(!a.attends(10, 5), "one before the half-window");
}

#[test]
fn a_causal_block_never_attends_forward() {
    let a = LayerAttention {
        rope_freq_base: 1_000_000.0,
        window: None,
        causal: true,
    };

    assert!(a.attends(10, 10));
    assert!(a.attends(10, 0));
    assert!(!a.attends(10, 11));
}

// ── Qwen3 ────────────────────────────────────────────────────────────────────

#[test]
fn qwen3_reads_the_explicit_head_dim_rather_than_deriving_it() {
    let cfg = EncoderConfig::from_gguf_metadata(&qwen3_metadata()).unwrap();

    assert_eq!(cfg.resolved_head_dim(), 128, "from attention.key_length");
    assert_eq!(
        cfg.head_dim(),
        64,
        "the derived formula gives a different answer, which is why it must not be used"
    );
    assert_ne!(cfg.resolved_head_dim(), cfg.head_dim());
}

#[test]
fn qwen3_is_causal_prenorm_gqa() {
    let cfg = EncoderConfig::from_gguf_metadata(&qwen3_metadata()).unwrap();

    assert_eq!(cfg.arch_family, ArchFamily::Qwen3);
    assert!(cfg.causal, "Qwen3-Embedding is a decoder backbone");
    assert_eq!(cfg.norm_scheme, NormScheme::PreNorm);
    assert_eq!(cfg.ffn_variant, FfnVariant::GatedSilu);
    assert_eq!(cfg.resolved_num_kv_heads(), 8);
    assert_eq!(cfg.num_attention_heads, 16);
    assert!(!cfg.embed_scale, "Qwen3 does not scale token embeddings");
    assert!(!cfg.interleaves_attention());
}

#[test]
fn qwen3_causal_blocks_carry_causality_into_the_layer_spec() {
    let cfg = EncoderConfig::from_gguf_metadata(&qwen3_metadata()).unwrap();
    let a = cfg.layer_attention(0);
    assert!(a.causal);
    assert_eq!(a.window, None);
}

#[test]
fn qwen3_rejects_a_pooling_type_it_was_not_trained_for() {
    let mut m = qwen3_metadata();
    m.kv.insert("qwen3.pooling_type".into(), GgufValue::Uint32(1));
    let err = EncoderConfig::from_gguf_metadata(&m)
        .unwrap_err()
        .to_string();
    assert!(err.contains("pooling_type"), "got: {err}");
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

#[test]
fn an_unsupported_architecture_is_named_in_the_error() {
    let m = meta(&[("general.architecture", GgufValue::String("mamba".into()))]);
    let err = EncoderConfig::from_gguf_metadata(&m)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("mamba"),
        "the error must name the architecture: {err}"
    );
    assert!(err.contains("qwen3"), "and list what is supported: {err}");
}

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

/// jina-bert-v3 must route to its own family, not to the BERT fallback: it
/// reports `XLMRobertaModel` upstream but has no `position_embd` tensor, and
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
