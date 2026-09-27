//! Reads oxidizr's `config.json` (its `BlazrConfig`, `oxidizr/src/config.rs:1165`,
//! written by `save_blazr_config_if_needed` at `oxidizr/src/trainer.rs:767`)
//! and translates it into boostr's [`UniversalConfig`] for
//! [`boostr::model::LoadedModel::load`].
//!
//! oxidizr's on-disk schema carries no `model_type` and no `rope_theta`: the
//! g2p_lm checkpoint is a plain dense-attention decoder (no Mamba, no MoE,
//! no MLA — `oxidizr/models/g2p-char/g2p-char.yaml`), so this module always
//! builds a `"llama"` `UniversalConfig` and fills `rope_theta` with the
//! training config's own value (10000.0, `default_rope_theta` in
//! `boostr/src/model/config/attention.rs`), which oxidizr does not persist.

use boostr::model::{AttentionConfig, UniversalConfig};
use serde::Deserialize;

use crate::error::{Error, Result};

/// The rope base oxidizr's on-disk config does not carry. Matches
/// `g2p-char.yaml`'s `model.rope_theta: 10000.0` and boostr's own default.
const G2P_LM_ROPE_THETA: f32 = 10000.0;

/// Fields oxidizr's `BlazrConfig::save` writes to `config.json`. Only the
/// ones a dense-attention checkpoint needs; the Mamba/MoE/MLA fields the
/// struct also writes are absent from a g2p_lm checkpoint and are not read
/// here.
#[derive(Debug, Deserialize)]
pub(crate) struct OxidizrCheckpointConfig {
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    rms_norm_eps: f64,
    max_seq_len: usize,
}

impl OxidizrCheckpointConfig {
    /// Parses `config.json`.
    pub(crate) fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| Error::ModelError {
            reason: format!("g2p_lm config.json does not parse: {e}"),
        })
    }

    /// Translates to boostr's `UniversalConfig`, model type `"llama"`.
    ///
    /// oxidizr's schema records no `num_kv_heads` (no GQA support in
    /// g2p_lm's dense-MHA training config) and no `head_dim` override, so
    /// both are left `None`: `LoadedModel::load` falls back to
    /// `num_kv_heads == num_heads` and `head_dim == hidden_size / num_heads`,
    /// which is exactly what `g2p-char.yaml` trained (MHA, 256 / 4 == 64).
    pub(crate) fn to_universal(&self) -> Result<UniversalConfig> {
        let num_heads = self.num_attention_heads.ok_or_else(|| Error::ModelError {
            reason: "g2p_lm config.json has no num_attention_heads".into(),
        })?;
        Ok(UniversalConfig {
            model_type: "llama".to_string(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            attention: Some(AttentionConfig {
                num_heads,
                num_kv_heads: None,
                head_dim: None,
                rope_theta: G2P_LM_ROPE_THETA,
                rope_scaling: None,
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                sliding_window: None,
                use_alibi: false,
            }),
            ssm: None,
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: self.tie_word_embeddings,
            grow_vocab: false,
            vision: None,
            audio: None,
            gdn: None,
            qwen35_attention: None,
            hadamard: None,
            quant_formats: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        r#"{
            "hidden_size": 256,
            "num_layers": 4,
            "vocab_size": 111,
            "tie_word_embeddings": true,
            "tokenizer_vocab": "custom",
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "rms_norm_eps": 1e-5,
            "max_seq_len": 128
        }"#
    }

    #[test]
    fn translates_a_dense_attention_checkpoint() {
        let cfg = OxidizrCheckpointConfig::from_json(sample()).unwrap_or_else(|e| panic!("{e}"));
        let universal = cfg.to_universal().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(universal.model_type, "llama");
        assert_eq!(universal.vocab_size, 111);
        assert_eq!(universal.hidden_size, 256);
        assert_eq!(universal.num_layers, 4);
        assert_eq!(universal.max_seq_len, 128);
        assert!(universal.tie_word_embeddings);
        let attn = universal
            .attention
            .unwrap_or_else(|| panic!("no attention config"));
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, None);
        assert_eq!(attn.rope_theta, G2P_LM_ROPE_THETA);
    }

    #[test]
    fn missing_num_attention_heads_is_an_error() {
        let bad = r#"{
            "hidden_size": 256, "num_layers": 4, "vocab_size": 111,
            "rms_norm_eps": 1e-5, "max_seq_len": 128
        }"#;
        let cfg = OxidizrCheckpointConfig::from_json(bad).unwrap_or_else(|e| panic!("{e}"));
        let err = cfg.to_universal().unwrap_err();
        assert!(err.to_string().contains("num_attention_heads"));
    }
}
