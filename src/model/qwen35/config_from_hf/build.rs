//! [`qwen35_config_from_hf`]: the `qwen35` [`UniversalConfig`] from a parsed
//! Hugging Face Qwen3.5 `config.json`.
//!
//! `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) nests the text
//! model under `text_config`. `Qwen3_5ForCausalLM` (`qwen3_5_text`) keeps
//! it at the top level. Each key is read from `text_config` first, then from
//! the top level, the merge order of the llama.cpp converter. RoPE keys are
//! read from `rope_parameters` first.
//!
//! | Key                                         | Field                            |
//! | ------------------------------------------- | -------------------------------- |
//! | `vocab_size`                                | `vocab_size`                     |
//! | `hidden_size`                               | `hidden_size`                    |
//! | `num_hidden_layers`                         | `num_layers`                     |
//! | `max_position_embeddings`                   | `max_seq_len`                    |
//! | `intermediate_size`                         | `intermediate_size`              |
//! | `rms_norm_eps` (`1e-6`)                     | `rms_norm_eps`, every `rms_eps`  |
//! | `tie_word_embeddings` (`false`)             | `tie_word_embeddings`            |
//! | `layer_types`, `full_attention_interval`    | `hybrid_layers`                  |
//! | `linear_conv_kernel_dim`                    | `gdn.conv_kernel`                |
//! | `linear_key_head_dim`                       | `gdn.state_size`                 |
//! | `linear_num_key_heads`                      | `gdn.key_heads`                  |
//! | `linear_num_value_heads`                    | `gdn.value_heads`                |
//! | `linear_num_value_heads * linear_value_head_dim` | `gdn.inner_size`            |
//! | `num_attention_heads`                       | `qwen35_attention.num_heads`     |
//! | `num_key_value_heads`                       | `qwen35_attention.num_kv_heads`  |
//! | `head_dim`                                  | `qwen35_attention.head_dim`      |
//! | `head_dim * partial_rotary_factor` (`0.25`) | `qwen35_attention.rope_dim`      |
//! | `mrope_section` (`[11, 11, 10]`), `0`-padded | `qwen35_attention.rope_sections` |
//! | `rope_theta`                                | `qwen35_attention.rope_theta`    |
//!
//! Defaults in parentheses are the converter's. With neither `layer_types`
//! nor `full_attention_interval`, every fourth layer is full attention.
//!
//! Rejected, each with the field and its value: `attn_output_gate` other
//! than `true`, `attention_bias` other than `false`, a non-empty
//! `mlp_only_layers`, `mrope_interleaved` other than `true`, `hidden_act`
//! other than `silu`, `rope_type` other than `default`, and
//! `linear_key_head_dim` different from `linear_value_head_dim`.
//!
//! Ignored: `mtp_*`, `mamba_ssm_dtype`, `vision_config`, dtypes and token ids.

use super::fields::{hybrid_layers, rope_dim, rope_sections};
use super::keys::{Keys, bad, check_supported, missing, opt_f64};
use crate::error::{Error, Result};
use crate::model::config::{
    GdnConfig, Qwen35AttentionConfig, UniversalConfig, default_gdn_chunk_size,
};
use serde_json::Value;

const ARCH: &str = "qwen35";

/// Top-level `model_type`s of Hugging Face Qwen3.5 dense checkpoints.
pub const QWEN35_HF_MODEL_TYPES: [&str; 2] = ["qwen3_5", "qwen3_5_text"];

/// `true` when `config`'s top-level `model_type` is in
/// [`QWEN35_HF_MODEL_TYPES`].
pub fn is_qwen35_hf_config(config: &Value) -> bool {
    config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|t| QWEN35_HF_MODEL_TYPES.contains(&t))
}

/// Build the `qwen35` [`UniversalConfig`] from a parsed Hugging Face
/// Qwen3.5 `config.json`. Text model only: `vision` stays `None`.
///
/// # Errors
///
/// [`Error::ModelError`] when `model_type` is not in
/// [`QWEN35_HF_MODEL_TYPES`], a required key is missing or has the wrong
/// type, an option the module docs list as rejected is set, the RoPE
/// sections do not sum to `rope_dim / 2`, or the assembled config fails
/// [`UniversalConfig::validate`].
pub fn qwen35_config_from_hf(config: &Value) -> Result<UniversalConfig> {
    let keys = Keys::new(config)?;
    if !is_qwen35_hf_config(config) {
        let model_type = config.get("model_type").cloned().unwrap_or(Value::Null);
        return Err(bad(
            "model_type",
            &model_type,
            "\"qwen3_5\" or \"qwen3_5_text\"",
        ));
    }
    check_supported(&keys)?;

    let num_layers = keys.usize("num_hidden_layers")?;
    let hidden_size = keys.usize("hidden_size")?;
    let num_heads = keys.usize("num_attention_heads")?;
    let head_dim = match keys.opt_usize("head_dim")? {
        Some(d) => d,
        None => hidden_size.checked_div(num_heads).unwrap_or(0),
    };
    let rms_eps = opt_f64("rms_norm_eps", keys.get("rms_norm_eps"))?.unwrap_or(1e-6) as f32;

    let value_heads = keys.usize("linear_num_value_heads")?;
    let gdn = GdnConfig {
        hidden_size,
        conv_kernel: keys.usize("linear_conv_kernel_dim")?,
        state_size: keys.usize("linear_key_head_dim")?,
        key_heads: keys.usize("linear_num_key_heads")?,
        value_heads,
        inner_size: value_heads * keys.usize("linear_value_head_dim")?,
        rms_eps,
        chunk_size: default_gdn_chunk_size(),
        v_grouped: false,
    };

    let rope_theta =
        opt_f64("rope_theta", keys.rope_get("rope_theta"))?.ok_or_else(|| missing("rope_theta"))?;
    let attention = Qwen35AttentionConfig {
        hidden_size,
        num_heads,
        num_kv_heads: keys.opt_usize("num_key_value_heads")?.unwrap_or(num_heads),
        head_dim,
        rope_dim: rope_dim(&keys, head_dim)?,
        rope_sections: rope_sections(&keys)?,
        rope_theta: rope_theta as f32,
        rms_eps,
    };
    let pairs: usize = attention.rope_sections.iter().sum();
    if pairs * 2 != attention.rope_dim {
        return Err(Error::ModelError {
            reason: format!(
                "Qwen3.5 config.json: mrope_section {:?} holds {pairs} rotated pairs; \
                 head_dim * partial_rotary_factor ({}) needs {}",
                attention.rope_sections,
                attention.rope_dim,
                attention.rope_dim / 2
            ),
        });
    }

    let tie_word_embeddings = match keys.get("tie_word_embeddings") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(v) => return Err(bad("tie_word_embeddings", v, "a boolean")),
    };

    let config = UniversalConfig {
        model_type: ARCH.to_string(),
        vocab_size: keys.usize("vocab_size")?,
        hidden_size,
        num_layers,
        max_seq_len: keys.usize("max_position_embeddings")?,
        intermediate_size: Some(keys.usize("intermediate_size")?),
        rms_norm_eps: f64::from(rms_eps),
        attention: None,
        ssm: None,
        moe: None,
        hybrid_layers: Some(hybrid_layers(&keys, num_layers)?),
        tie_word_embeddings,
        grow_vocab: false,
        vision: None,
        audio: None,
        gdn: Some(gdn),
        qwen35_attention: Some(attention),
        hadamard: None,
        block: Default::default(),
        quant_formats: Vec::new(),
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{CLEF_FLASH, QWEN35_4B_BASE, QWEN35_9B, parse};
    use super::*;

    /// The fields every fixture shares, beyond width, FFN and tie.
    fn check_common(cfg: &UniversalConfig) {
        assert_eq!(cfg.model_type, "qwen35");
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.max_seq_len, 262_144);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(cfg.vision.is_none() && cfg.hadamard.is_none());

        let gdn = cfg.gdn.as_ref().unwrap();
        assert_eq!(gdn.hidden_size, cfg.hidden_size);
        assert_eq!(gdn.conv_kernel, 4);
        assert_eq!(gdn.state_size, 128);
        assert_eq!(gdn.key_heads, 16);
        assert_eq!(gdn.value_heads, 32);
        assert_eq!(gdn.inner_size, 4096);
        assert!(!gdn.v_grouped);

        let attn = cfg.qwen35_attention.as_ref().unwrap();
        assert_eq!(attn.num_heads, 16);
        assert_eq!(attn.num_kv_heads, 4);
        assert_eq!(attn.head_dim, 256);
        assert_eq!(attn.rope_dim, 64);
        assert_eq!(attn.rope_sections, [11, 11, 10, 0]);
        assert_eq!(attn.rope_theta, 1e7);

        let layers = cfg.hybrid_layers.as_ref().unwrap();
        let want: Vec<usize> = (0..32).filter(|i| (i + 1) % 4 == 0).collect();
        assert_eq!(layers.attention_layers, want);
        assert_eq!(layers.ssm_layers.len(), 24);
    }

    #[test]
    fn qwen35_4b_base_is_tied() {
        let cfg = qwen35_config_from_hf(&parse(QWEN35_4B_BASE)).unwrap();
        check_common(&cfg);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, Some(9216));
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn qwen35_9b_reads_the_tie_flag_from_the_top_level() {
        let cfg = qwen35_config_from_hf(&parse(QWEN35_9B)).unwrap();
        check_common(&cfg);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, Some(12_288));
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn clef_flash_derives_the_9b_geometry() {
        let cfg = qwen35_config_from_hf(&parse(CLEF_FLASH)).unwrap();
        check_common(&cfg);
        assert_eq!(cfg.hidden_size, 4096);
        assert!(!cfg.tie_word_embeddings);
    }

    /// `Qwen3_5ForCausalLM`: the text keys at the top level.
    #[test]
    fn text_only_config_reads_top_level_keys() {
        let mut text = parse(QWEN35_9B)["text_config"].clone();
        text["tie_word_embeddings"] = Value::Bool(false);
        assert!(is_qwen35_hf_config(&text));
        check_common(&qwen35_config_from_hf(&text).unwrap());
    }

    #[test]
    fn config_file_loaders_take_the_qwen35_branch() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, parse(QWEN35_9B).to_string()).unwrap();
        check_common(&crate::model::config::load_config_auto(&path).unwrap());
        check_common(&crate::model::config::load_huggingface_config(&path).unwrap());
    }

    /// The 9B config with `text_config.<key>` (or the `rope_parameters`
    /// entry) set to `value`, and the error text.
    fn reject(key: &str, value: Value, rope: bool) -> String {
        let mut json = parse(QWEN35_9B);
        let block = if rope {
            &mut json["text_config"]["rope_parameters"]
        } else {
            &mut json["text_config"]
        };
        block[key] = value;
        qwen35_config_from_hf(&json).unwrap_err().to_string()
    }

    #[test]
    fn rejected_options_name_the_field_and_value() {
        let cases = [
            ("attn_output_gate", Value::Bool(false), false, "false"),
            ("attention_bias", Value::Bool(true), false, "true"),
            ("mlp_only_layers", serde_json::json!([0, 4]), false, "[0,4]"),
            ("mrope_interleaved", Value::Bool(false), true, "false"),
            ("hidden_act", Value::from("gelu"), false, "\"gelu\""),
            ("rope_type", Value::from("yarn"), true, "\"yarn\""),
            ("linear_value_head_dim", Value::from(64), false, "64"),
        ];
        for (key, value, rope, shown) in cases {
            let err = reject(key, value, rope);
            assert!(err.contains(key) && err.contains(shown), "{key}: {err}");
        }
    }

    #[test]
    fn layer_types_must_agree_with_the_interval() {
        let err = reject("full_attention_interval", Value::from(2), false);
        assert!(err.contains("full_attention_interval 2"), "{err}");
    }

    #[test]
    fn mrope_sections_must_cover_the_rotated_width() {
        let err = reject("partial_rotary_factor", Value::from(0.5), true);
        assert!(err.contains("mrope_section"), "{err}");
    }

    #[test]
    fn missing_key_and_other_model_types_are_errors() {
        let mut json = parse(QWEN35_9B);
        json["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("vocab_size");
        let err = qwen35_config_from_hf(&json).unwrap_err().to_string();
        assert!(err.contains("vocab_size"), "{err}");

        let mut json = parse(QWEN35_9B);
        json["model_type"] = Value::from("qwen3_5_moe");
        assert!(!is_qwen35_hf_config(&json));
        let err = qwen35_config_from_hf(&json).unwrap_err().to_string();
        assert!(err.contains("qwen3_5_moe"), "{err}");
    }
}
