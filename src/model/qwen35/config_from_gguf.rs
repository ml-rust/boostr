//! [`UniversalConfig`] from the `qwen35.*` GGUF namespace.
//!
//! Key table, matching `llama_model_base::load_hparams`:
//!
//! | GGUF key                                  | Field                          |
//! | ----------------------------------------- | ------------------------------ |
//! | `qwen35.block_count`                      | `num_layers`                   |
//! | `qwen35.context_length`                   | `max_seq_len`                  |
//! | `qwen35.embedding_length`                 | `hidden_size`                  |
//! | `qwen35.feed_forward_length`              | `intermediate_size`            |
//! | `qwen35.attention.head_count`             | `qwen35_attention.num_heads`   |
//! | `qwen35.attention.head_count_kv`          | `qwen35_attention.num_kv_heads`|
//! | `qwen35.attention.key_length`             | `qwen35_attention.head_dim`    |
//! | `qwen35.attention.layer_norm_rms_epsilon` | `rms_norm_eps`, both `rms_eps` |
//! | `qwen35.rope.dimension_count`             | `qwen35_attention.rope_dim`    |
//! | `qwen35.rope.dimension_sections`          | `qwen35_attention.rope_sections`|
//! | `qwen35.rope.freq_base`                   | `qwen35_attention.rope_theta`  |
//! | `qwen35.ssm.conv_kernel`                  | `gdn.conv_kernel`              |
//! | `qwen35.ssm.state_size`                   | `gdn.state_size`               |
//! | `qwen35.ssm.group_count`                  | `gdn.key_heads`                |
//! | `qwen35.ssm.time_step_rank`               | `gdn.value_heads`              |
//! | `qwen35.ssm.inner_size`                   | `gdn.inner_size`               |
//! | `qwen35.full_attention_interval`          | `hybrid_layers`                |
//! | `tokenizer.ggml.tokens` (length)          | `vocab_size`                   |
//! | `prism.hadamard.*`                        | `hadamard`, `gdn.v_grouped`    |

use crate::error::{Error, Result};
use crate::format::gguf::{GgufMetadata, HadamardContract};
use crate::model::config::{
    GdnConfig, HybridConfig, Qwen35AttentionConfig, UniversalConfig, default_gdn_chunk_size,
};

const ARCH: &str = "qwen35";

fn missing(key: &str) -> Error {
    Error::ModelError {
        reason: format!("GGUF missing {key}"),
    }
}

fn required_usize(meta: &GgufMetadata, key: &str) -> Result<usize> {
    meta.get_u32(key)
        .map(|v| v as usize)
        .ok_or_else(|| missing(key))
}

fn required_f32(meta: &GgufMetadata, key: &str) -> Result<f32> {
    meta.get_f32(key).ok_or_else(|| missing(key))
}

/// `tokenizer.ggml.tokens` length, else `qwen35.vocab_size`.
fn vocab_size(meta: &GgufMetadata) -> Result<usize> {
    if let Some(tokens) = meta.get_array("tokenizer.ggml.tokens") {
        return Ok(tokens.len());
    }
    required_usize(meta, "qwen35.vocab_size")
        .map_err(|_| missing("tokenizer.ggml.tokens (or qwen35.vocab_size)"))
}

fn rope_sections(meta: &GgufMetadata) -> Result<[usize; 4]> {
    const KEY: &str = "qwen35.rope.dimension_sections";
    let raw = meta.get_i64_array(KEY).ok_or_else(|| missing(KEY))?;
    if raw.len() != 4 || raw.iter().any(|v| *v < 0) {
        return Err(Error::ModelError {
            reason: format!("{KEY} must be 4 non-negative ints, got {raw:?}"),
        });
    }
    Ok([
        raw[0] as usize,
        raw[1] as usize,
        raw[2] as usize,
        raw[3] as usize,
    ])
}

/// Build the `qwen35` [`UniversalConfig`] from GGUF metadata.
///
/// # Errors
///
/// [`Error::ModelError`] when `general.architecture` is not `qwen35`, a
/// key in the module table is missing, `attention.value_length` differs
/// from `attention.key_length`, the `prism.hadamard.*` block is malformed,
/// or the assembled config fails [`UniversalConfig::validate`].
pub fn qwen35_config_from_gguf(meta: &GgufMetadata) -> Result<UniversalConfig> {
    let arch = meta
        .architecture()
        .ok_or_else(|| missing("general.architecture"))?;
    if arch != ARCH {
        return Err(Error::ModelError {
            reason: format!("general.architecture is '{arch}', expected '{ARCH}'"),
        });
    }

    let num_layers = required_usize(meta, "qwen35.block_count")?;
    let max_seq_len = required_usize(meta, "qwen35.context_length")?;
    let hidden_size = required_usize(meta, "qwen35.embedding_length")?;
    let intermediate_size = required_usize(meta, "qwen35.feed_forward_length")?;
    let rms_eps = required_f32(meta, "qwen35.attention.layer_norm_rms_epsilon")?;

    let head_dim = required_usize(meta, "qwen35.attention.key_length")?;
    let value_length = required_usize(meta, "qwen35.attention.value_length")?;
    if value_length != head_dim {
        return Err(Error::ModelError {
            reason: format!(
                "qwen35.attention.value_length ({value_length}) != key_length ({head_dim})"
            ),
        });
    }

    let hadamard = HadamardContract::from_metadata(meta)?;

    let gdn = GdnConfig {
        hidden_size,
        conv_kernel: required_usize(meta, "qwen35.ssm.conv_kernel")?,
        state_size: required_usize(meta, "qwen35.ssm.state_size")?,
        key_heads: required_usize(meta, "qwen35.ssm.group_count")?,
        value_heads: required_usize(meta, "qwen35.ssm.time_step_rank")?,
        inner_size: required_usize(meta, "qwen35.ssm.inner_size")?,
        rms_eps,
        chunk_size: default_gdn_chunk_size(),
        v_grouped: hadamard.as_ref().is_some_and(|h| h.gdn_v_grouped),
    };

    let attention = Qwen35AttentionConfig {
        hidden_size,
        num_heads: required_usize(meta, "qwen35.attention.head_count")?,
        num_kv_heads: required_usize(meta, "qwen35.attention.head_count_kv")?,
        head_dim,
        rope_dim: required_usize(meta, "qwen35.rope.dimension_count")?,
        rope_sections: rope_sections(meta)?,
        rope_theta: required_f32(meta, "qwen35.rope.freq_base")?,
        rms_eps,
    };

    let interval = required_usize(meta, "qwen35.full_attention_interval")?;

    let config = UniversalConfig {
        model_type: ARCH.to_string(),
        vocab_size: vocab_size(meta)?,
        hidden_size,
        num_layers,
        max_seq_len,
        intermediate_size: Some(intermediate_size),
        rms_norm_eps: rms_eps as f64,
        attention: None,
        ssm: None,
        moe: None,
        hybrid_layers: Some(HybridConfig::from_full_attention_interval(
            num_layers, interval,
        )),
        // The Bonsai file ships a separate `output.weight`. The loader
        // errors when this is set and the head is absent.
        tie_word_embeddings: false,
        grow_vocab: false,
        vision: None,
        audio: None,
        gdn: Some(gdn),
        qwen35_attention: Some(attention),
        hadamard,
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::GgufValue;

    fn u32v(v: u32) -> GgufValue {
        GgufValue::Uint32(v)
    }

    /// The Bonsai-2-27B header, minus the tokenizer array (replaced by
    /// `qwen35.vocab_size`) and the `prism.hadamard.*` block.
    fn bonsai_pairs() -> Vec<(&'static str, GgufValue)> {
        vec![
            ("general.architecture", GgufValue::String(ARCH.into())),
            ("qwen35.block_count", u32v(64)),
            ("qwen35.context_length", u32v(262_144)),
            ("qwen35.embedding_length", u32v(5120)),
            ("qwen35.feed_forward_length", u32v(17_408)),
            ("qwen35.attention.head_count", u32v(24)),
            ("qwen35.attention.head_count_kv", u32v(4)),
            ("qwen35.attention.key_length", u32v(256)),
            ("qwen35.attention.value_length", u32v(256)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(1e-6),
            ),
            ("qwen35.rope.dimension_count", u32v(64)),
            (
                "qwen35.rope.dimension_sections",
                GgufValue::Array(vec![
                    GgufValue::Int32(11),
                    GgufValue::Int32(11),
                    GgufValue::Int32(10),
                    GgufValue::Int32(0),
                ]),
            ),
            ("qwen35.rope.freq_base", GgufValue::Float32(1e7)),
            ("qwen35.ssm.conv_kernel", u32v(4)),
            ("qwen35.ssm.state_size", u32v(128)),
            ("qwen35.ssm.group_count", u32v(16)),
            ("qwen35.ssm.time_step_rank", u32v(48)),
            ("qwen35.ssm.inner_size", u32v(6144)),
            ("qwen35.full_attention_interval", u32v(4)),
            ("qwen35.vocab_size", u32v(248_320)),
        ]
    }

    fn meta(pairs: Vec<(&str, GgufValue)>) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        for (k, v) in pairs {
            m.kv.insert(k.to_string(), v);
        }
        m
    }

    #[test]
    fn bonsai_header_derives_full_config() {
        let cfg = qwen35_config_from_gguf(&meta(bonsai_pairs())).unwrap();
        assert_eq!(cfg.model_type, "qwen35");
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_layers, 64);
        assert_eq!(cfg.max_seq_len, 262_144);
        assert_eq!(cfg.intermediate_size, Some(17_408));
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.hadamard.is_none());

        let gdn = cfg.gdn.as_ref().unwrap();
        assert_eq!(gdn.conv_kernel, 4);
        assert_eq!(gdn.state_size, 128);
        assert_eq!(gdn.key_heads, 16);
        assert_eq!(gdn.value_heads, 48);
        assert_eq!(gdn.inner_size, 6144);
        assert_eq!(gdn.qkv_dim(), 10_240);
        assert!(!gdn.v_grouped);

        let attn = cfg.qwen35_attention.as_ref().unwrap();
        assert_eq!(attn.num_heads, 24);
        assert_eq!(attn.num_kv_heads, 4);
        assert_eq!(attn.head_dim, 256);
        assert_eq!(attn.rope_dim, 64);
        assert_eq!(attn.rope_sections, [11, 11, 10, 0]);
        assert_eq!(attn.rope_theta, 1e7);
        assert_eq!(attn.q_gate_dim(), 12_288);

        let layers = cfg.hybrid_layers.as_ref().unwrap();
        let want: Vec<usize> = (0..64).filter(|i| (i + 1) % 4 == 0).collect();
        assert_eq!(layers.attention_layers, want);
        assert_eq!(layers.attention_layers.len(), 16);
        assert_eq!(layers.ssm_layers.len(), 48);
    }

    #[test]
    fn vocab_size_prefers_token_array() {
        let mut pairs = bonsai_pairs();
        pairs.push((
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![GgufValue::String("a".into()); 7]),
        ));
        let cfg = qwen35_config_from_gguf(&meta(pairs)).unwrap();
        assert_eq!(cfg.vocab_size, 7);
    }

    #[test]
    fn missing_key_names_it() {
        let pairs: Vec<_> = bonsai_pairs()
            .into_iter()
            .filter(|(k, _)| *k != "qwen35.ssm.group_count")
            .collect();
        let err = qwen35_config_from_gguf(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains("qwen35.ssm.group_count"), "{err}");
    }

    #[test]
    fn wrong_architecture_errors() {
        let mut pairs = bonsai_pairs();
        pairs.retain(|(k, _)| *k != "general.architecture");
        pairs.push(("general.architecture", GgufValue::String("llama".into())));
        let err = qwen35_config_from_gguf(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains("llama"), "{err}");
    }

    #[test]
    fn value_length_mismatch_errors() {
        let mut pairs = bonsai_pairs();
        pairs.retain(|(k, _)| *k != "qwen35.attention.value_length");
        pairs.push(("qwen35.attention.value_length", u32v(128)));
        assert!(qwen35_config_from_gguf(&meta(pairs)).is_err());
    }
}
