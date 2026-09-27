//! Reads a g2p_lm checkpoint's `config.json`: boostr's [`UniversalConfig`],
//! which oxidizr writes beside every checkpoint it saves.

use boostr::model::UniversalConfig;

use crate::error::{Error, Result};

/// Parses `config.json` and checks it with [`UniversalConfig::validate`].
pub(crate) fn parse_checkpoint_config(text: &str) -> Result<UniversalConfig> {
    let config: UniversalConfig = serde_json::from_str(text).map_err(|e| Error::ModelError {
        reason: format!("g2p_lm config.json does not parse: {e}"),
    })?;
    config.validate().map_err(|e| Error::ModelError {
        reason: format!("g2p_lm config.json is invalid: {e}"),
    })?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "model_type": "llama",
        "vocab_size": 111,
        "hidden_size": 256,
        "num_layers": 4,
        "max_seq_len": 128,
        "intermediate_size": 1024,
        "rms_norm_eps": 0.00001,
        "attention": {
            "num_heads": 4,
            "num_kv_heads": null,
            "head_dim": 64,
            "rope_theta": 10000.0,
            "rope_scaling": null,
            "kv_latent_dim": null,
            "q_latent_dim": null,
            "d_rope": null,
            "sliding_window": null,
            "use_alibi": false
        },
        "ssm": null,
        "moe": null,
        "hybrid_layers": null,
        "tie_word_embeddings": true,
        "grow_vocab": false,
        "vision": null,
        "audio": null,
        "gdn": null,
        "qwen35_attention": null,
        "hadamard": null
    }"#;

    #[test]
    fn parses_the_config_oxidizr_writes() {
        let config = parse_checkpoint_config(SAMPLE).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.vocab_size, 111);
        assert_eq!(config.num_layers, 4);
        assert!(config.tie_word_embeddings);
        let attn = config
            .attention
            .unwrap_or_else(|| panic!("no attention config"));
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.rope_theta, 10000.0);
    }

    #[test]
    fn rejects_a_config_that_does_not_parse() {
        let err = parse_checkpoint_config(r#"{"hidden_size": 256}"#).unwrap_err();
        assert!(err.to_string().contains("does not parse"), "{err}");
    }

    #[test]
    fn rejects_a_config_that_fails_validation() {
        let zero_vocab = SAMPLE.replace("\"vocab_size\": 111", "\"vocab_size\": 0");
        let err = parse_checkpoint_config(&zero_vocab).unwrap_err();
        assert!(err.to_string().contains("invalid"), "{err}");
    }
}
