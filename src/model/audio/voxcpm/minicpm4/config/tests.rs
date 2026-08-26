//! Unit tests for [`MiniCpm4Config`] parsing and the `residual_lm`
//! overrides.
//!
//! In a sibling file rather than inline so `config.rs` stays inside the
//! file-size limit — the same split `decode.rs` uses.

use super::*;

/// `lm_config` body with `head_dim` (`kv_channels`) 4, so the RoPE factor
/// lists stay short enough to write out.
fn config_json(extra: &str) -> String {
    format!(
        r#"{{"lm_config":{{
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "kv_channels": 4,
            "vocab_size": 100,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
            "rope_scaling": {{
                "short_factor": [1.0, 2.0],
                "long_factor": [3.0, 4.0],
                "original_max_position_embeddings": 256
            }}{extra}
        }}}}"#
    )
}

/// The same `lm_config` body plus the two top-level `residual_lm_*` keys
/// the real checkpoint carries alongside it.
fn residual_config_json(extra: &str) -> String {
    let inner = config_json("");
    let inner = inner
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .expect("config_json is a JSON object");
    format!("{{{inner},{extra}}}")
}

fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, body).expect("write temp config");
    path
}

#[test]
fn default_head_dim_matches_short_factor_len() {
    let cfg = MiniCpm4Config::default();
    assert_eq!(cfg.rope_short_factor.len(), cfg.head_dim / 2);
    assert_eq!(cfg.rope_long_factor.len(), cfg.head_dim / 2);
    assert!(cfg.has_embedding());
}

#[test]
fn parses_lm_config_section() {
    let path = write_temp("boostr_minicpm4_ok.json", &config_json(""));
    let cfg = MiniCpm4Config::from_config_json(&path).expect("parse");
    let _ = std::fs::remove_file(&path);

    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.hidden_size, 8);
    assert_eq!(cfg.intermediate_size, 16);
    assert_eq!(cfg.num_heads, 4);
    assert_eq!(cfg.num_kv_heads, 2);
    // head_dim comes from kv_channels (4), NOT hidden_size/num_heads (2).
    assert_eq!(cfg.head_dim, 4);
    assert_eq!(cfg.vocab_size, 100);
    assert_eq!(cfg.max_position_embeddings, 512);
    assert_eq!(cfg.original_max_position_embeddings, 256);
    assert_eq!(cfg.rope_short_factor, vec![1.0, 2.0]);
    assert_eq!(cfg.rope_long_factor, vec![3.0, 4.0]);
}

#[test]
fn zero_vocab_has_no_embedding() {
    let body = config_json("").replace("\"vocab_size\": 100", "\"vocab_size\": 0");
    let path = write_temp("boostr_minicpm4_novocab.json", &body);
    let cfg = MiniCpm4Config::from_config_json(&path).expect("parse");
    let _ = std::fs::remove_file(&path);
    assert_eq!(cfg.vocab_size, 0);
    assert!(!cfg.has_embedding());
}

#[test]
fn rejects_use_mup() {
    let path = write_temp(
        "boostr_minicpm4_mup.json",
        &config_json(",\n\"use_mup\": true"),
    );
    let err = MiniCpm4Config::from_config_json(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("use_mup"), "got {err}");
}

#[test]
fn rejects_short_factor_length_mismatch() {
    let body = config_json("").replace("\"short_factor\": [1.0, 2.0]", "\"short_factor\": [1.0]");
    let path = write_temp("boostr_minicpm4_badrope.json", &body);
    let err = MiniCpm4Config::from_config_json(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("short_factor"), "got {err}");
}

#[test]
fn rejects_missing_section() {
    let path = write_temp("boostr_minicpm4_nosection.json", &config_json(""));
    let err = MiniCpm4Config::from_config_json_section(&path, "residual_lm_config").unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("residual_lm_config"), "got {err}");
}

#[test]
fn rejects_missing_file() {
    assert!(MiniCpm4Config::from_config_json("/nonexistent/config.json").is_err());
}

#[test]
fn base_lm_section_is_not_nope() {
    let path = write_temp("boostr_minicpm4_baserope.json", &config_json(""));
    let cfg = MiniCpm4Config::from_config_json(&path).expect("parse");
    let _ = std::fs::remove_file(&path);
    assert!(!cfg.no_rope);
    assert!(cfg.uses_rope());
}

#[test]
fn residual_lm_applies_the_three_overrides() {
    let body = residual_config_json("\"residual_lm_num_layers\": 8, \"residual_lm_no_rope\": true");
    let path = write_temp("boostr_minicpm4_residual.json", &body);
    let cfg = MiniCpm4Config::residual_lm_from_config_json(&path).expect("parse");
    let base = MiniCpm4Config::from_config_json(&path).expect("parse");
    let _ = std::fs::remove_file(&path);

    // The three overrides.
    assert_eq!(cfg.num_layers, 8);
    assert_eq!(cfg.vocab_size, 0);
    assert!(cfg.no_rope);
    assert!(!cfg.uses_rope());
    assert!(!cfg.has_embedding());
    // Everything else is `lm_config` verbatim, including the RoPE tables
    // that a NoPE stack never reads.
    assert_eq!(cfg.hidden_size, base.hidden_size);
    assert_eq!(cfg.num_heads, base.num_heads);
    assert_eq!(cfg.num_kv_heads, base.num_kv_heads);
    assert_eq!(cfg.head_dim, base.head_dim);
    assert_eq!(cfg.rope_short_factor, base.rope_short_factor);
    // The base config it was derived from is untouched.
    assert_eq!(base.num_layers, 2);
    assert_eq!(base.vocab_size, 100);
    assert!(!base.no_rope);
}

#[test]
fn residual_lm_honours_a_false_no_rope() {
    let body =
        residual_config_json("\"residual_lm_num_layers\": 3, \"residual_lm_no_rope\": false");
    let path = write_temp("boostr_minicpm4_residual_rope.json", &body);
    let cfg = MiniCpm4Config::residual_lm_from_config_json(&path).expect("parse");
    let _ = std::fs::remove_file(&path);
    assert_eq!(cfg.num_layers, 3);
    assert!(!cfg.no_rope);
}

#[test]
fn residual_lm_rejects_missing_no_rope_key() {
    let body = residual_config_json("\"residual_lm_num_layers\": 8");
    let path = write_temp("boostr_minicpm4_residual_norope_key.json", &body);
    let err = MiniCpm4Config::residual_lm_from_config_json(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("residual_lm_no_rope"), "got {err}");
}

#[test]
fn residual_lm_rejects_missing_num_layers_key() {
    let body = residual_config_json("\"residual_lm_no_rope\": true");
    let path = write_temp("boostr_minicpm4_residual_nolayers_key.json", &body);
    let err = MiniCpm4Config::residual_lm_from_config_json(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("residual_lm_num_layers"),
        "got {err}"
    );
}
