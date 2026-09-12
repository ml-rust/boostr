//! `config.json` entry points for [`MiniCpm4Config`], and the raw
//! HuggingFace-spelled schema they deserialize through.

use super::schema::{
    DEFAULT_CONFIG_SECTION, MiniCpm4Config, RESIDUAL_LM_NO_ROPE_KEY, RESIDUAL_LM_NUM_LAYERS_KEY,
};
use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

impl MiniCpm4Config {
    /// Parse `lm_config` out of a VoxCPM2 `config.json` — i.e. `base_lm`.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_config_json_section(path, DEFAULT_CONFIG_SECTION)
    }

    /// Parse an arbitrary sub-object of a VoxCPM2 `config.json`.
    ///
    /// `section` is a top-level key (`"lm_config"` for `base_lm`). It is a
    /// parameter so a checkpoint that DOES carry a second architecture
    /// sub-object needs no second parser.
    ///
    /// This does NOT reach `residual_lm`: that checkpoint has no
    /// `residual_lm_config` section to name. Use
    /// [`residual_lm_from_config_json`](Self::residual_lm_from_config_json).
    pub fn from_config_json_section<P: AsRef<Path>>(path: P, section: &str) -> Result<Self> {
        Self::from_root(&read_config_root(path)?, section)
    }

    /// Parse `lm_config` out of the VERBATIM CONTENTS of a `config.json`.
    ///
    /// Split from [`from_config_json`](Self::from_config_json) so a container
    /// that carries the config as a string rather than a file — a GGUF's
    /// `voxcpm2.config_json` metadata key — runs through exactly this parse
    /// and every validation the file path applies, `use_mup` included.
    pub fn from_config_str(content: &str) -> Result<Self> {
        Self::from_root(&parse_config_root(content)?, DEFAULT_CONFIG_SECTION)
    }

    /// Resolve `residual_lm`'s config from a VoxCPM2 `config.json`.
    ///
    /// `residual_lm` has no sub-object of its own. The reference deep-copies
    /// `lm_config` and overrides exactly three fields
    /// (`voxcpm2.py:189-193`), and this reproduces that:
    ///
    /// | field | value |
    /// | --- | --- |
    /// | `num_layers` | top-level `residual_lm_num_layers` |
    /// | `no_rope` | top-level `residual_lm_no_rope` |
    /// | `vocab_size` | `0` — no `embed_tokens`, no `lm_head` |
    ///
    /// Both top-level keys are REQUIRED. Defaulting a missing
    /// `residual_lm_no_rope` to `false` would silently rotate a stack the
    /// reference never rotates, staying shape-valid while computing a
    /// different model.
    pub fn residual_lm_from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::residual_lm_from_root(&read_config_root(path)?)
    }

    /// Resolve `residual_lm`'s config from the VERBATIM CONTENTS of a
    /// `config.json` — the string sibling of
    /// [`residual_lm_from_config_json`](Self::residual_lm_from_config_json),
    /// for a GGUF's `voxcpm2.config_json` metadata key. Both top-level keys
    /// stay REQUIRED here for the same reason.
    pub fn residual_lm_from_config_str(content: &str) -> Result<Self> {
        Self::residual_lm_from_root(&parse_config_root(content)?)
    }

    /// The shared `residual_lm` resolution, once the root JSON is parsed.
    fn residual_lm_from_root(root: &serde_json::Value) -> Result<Self> {
        let base = Self::from_root(root, DEFAULT_CONFIG_SECTION)?;
        let num_layers = root
            .get(RESIDUAL_LM_NUM_LAYERS_KEY)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "VoxCPM2 config.json has no integer `{RESIDUAL_LM_NUM_LAYERS_KEY}`; \
                     residual_lm has no config section of its own, so its layer count \
                     can only come from that top-level key"
                ),
            })? as usize;
        let no_rope = root
            .get(RESIDUAL_LM_NO_ROPE_KEY)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "VoxCPM2 config.json has no boolean `{RESIDUAL_LM_NO_ROPE_KEY}`; \
                     guessing it would silently apply or drop RoPE"
                ),
            })?;
        Ok(base.into_residual_lm(num_layers, no_rope))
    }

    /// Resolve one architecture sub-object out of an already-parsed root.
    fn from_root(root: &serde_json::Value, section: &str) -> Result<Self> {
        let sub = root.get(section).ok_or_else(|| Error::ModelError {
            reason: format!("VoxCPM2 config.json has no `{section}` object"),
        })?;
        let raw: RawLmConfig =
            serde_json::from_value(sub.clone()).map_err(|e| Error::ModelError {
                reason: format!("invalid VoxCPM2 config.json `{section}`: {e}"),
            })?;
        raw.resolve()
    }
}

/// Read and parse a VoxCPM2 `config.json` once.
fn read_config_root<P: AsRef<Path>>(path: P) -> Result<serde_json::Value> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
        reason: format!("failed to read {}: {e}", path.as_ref().display()),
    })?;
    parse_config_root(&content)
}

/// Parse an already-read VoxCPM2 `config.json` body.
fn parse_config_root(content: &str) -> Result<serde_json::Value> {
    serde_json::from_str(content).map_err(|e| Error::ModelError {
        reason: format!("invalid VoxCPM2 config.json: {e}"),
    })
}

/// One MiniCPM4 config sub-object.
///
/// Field names are the HuggingFace spellings, read verbatim from the
/// checkpoint's `lm_config`: `num_hidden_layers`, `hidden_size`,
/// `intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
/// `kv_channels`, `vocab_size`, `rms_norm_eps`, `rope_theta`,
/// `max_position_embeddings`, `use_mup`. `original_max_position_embeddings`
/// lives inside the nested `rope_scaling` object, not alongside them.
#[derive(Debug, Deserialize)]
struct RawLmConfig {
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    /// Per-head width. Distinct from `hidden_size / num_attention_heads` —
    /// see [`MiniCpm4Config::head_dim`].
    kv_channels: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
    rope_scaling: RawRopeScaling,
    /// muP parameterization. `false` on this checkpoint. When it is `true`
    /// the decoder residual carries a `scale_depth / sqrt(num_layers)`
    /// factor and the embedding output is multiplied by `scale_emb`, none of
    /// which this port implements — so a `true` here is rejected rather than
    /// silently ignored.
    #[serde(default)]
    use_mup: bool,
}

#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
}

impl RawLmConfig {
    fn resolve(self) -> Result<MiniCpm4Config> {
        if self.use_mup {
            return Err(Error::ModelError {
                reason: "use_mup=true is not supported: the muP residual scaling \
                         (scale_depth/sqrt(num_layers)) and scale_emb are not \
                         implemented, and ignoring them computes a different model"
                    .to_string(),
            });
        }
        let half_dim = self.kv_channels / 2;
        if self.rope_scaling.short_factor.len() != half_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_scaling.short_factor has {} entries, expected {half_dim} \
                     (kv_channels/2)",
                    self.rope_scaling.short_factor.len()
                ),
            });
        }
        if self.rope_scaling.long_factor.len() != half_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_scaling.long_factor has {} entries, expected {half_dim} \
                     (kv_channels/2)",
                    self.rope_scaling.long_factor.len()
                ),
            });
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::ModelError {
                reason: format!(
                    "num_attention_heads ({}) must be a nonzero multiple of \
                     num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            });
        }
        Ok(MiniCpm4Config {
            num_layers: self.num_hidden_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.kv_channels,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            original_max_position_embeddings: self
                .rope_scaling
                .original_max_position_embeddings
                .unwrap_or(self.max_position_embeddings),
            rope_short_factor: self.rope_scaling.short_factor,
            rope_long_factor: self.rope_scaling.long_factor,
            // `lm_config` carries no `no_rope`: NoPE is a `residual_lm`
            // override applied by `into_residual_lm`, never a parsed field.
            no_rope: false,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`MiniCpm4Config`] parsing and the `residual_lm`
    //! overrides.

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
        let body =
            config_json("").replace("\"short_factor\": [1.0, 2.0]", "\"short_factor\": [1.0]");
        let path = write_temp("boostr_minicpm4_badrope.json", &body);
        let err = MiniCpm4Config::from_config_json(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(err.to_string().contains("short_factor"), "got {err}");
    }

    #[test]
    fn rejects_missing_section() {
        let path = write_temp("boostr_minicpm4_nosection.json", &config_json(""));
        let err =
            MiniCpm4Config::from_config_json_section(&path, "residual_lm_config").unwrap_err();
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
        let body =
            residual_config_json("\"residual_lm_num_layers\": 8, \"residual_lm_no_rope\": true");
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

    /// The string entry point parses the same body the file entry point does,
    /// and lands on the same config — a GGUF's embedded `config.json` is not a
    /// second parser.
    #[test]
    fn from_config_str_matches_from_config_json() {
        let body = config_json("");
        let path = write_temp("boostr_minicpm4_str.json", &body);
        let from_file = MiniCpm4Config::from_config_json(&path).expect("parse");
        let _ = std::fs::remove_file(&path);
        let from_str = MiniCpm4Config::from_config_str(&body).expect("parse");
        assert_eq!(from_str.num_layers, from_file.num_layers);
        assert_eq!(from_str.hidden_size, from_file.hidden_size);
        assert_eq!(from_str.head_dim, from_file.head_dim);
        assert_eq!(from_str.vocab_size, from_file.vocab_size);
        assert_eq!(from_str.rope_short_factor, from_file.rope_short_factor);
        assert_eq!(from_str.rope_long_factor, from_file.rope_long_factor);
    }

    /// The reference-parity trap the file path guards must still fire on the
    /// string path: muP is REJECTED, never ignored.
    #[test]
    fn from_config_str_rejects_use_mup() {
        let err =
            MiniCpm4Config::from_config_str(&config_json(",\n\"use_mup\": true")).unwrap_err();
        assert!(err.to_string().contains("use_mup"), "got {err}");
    }

    #[test]
    fn from_config_str_rejects_missing_section_and_bad_json() {
        assert!(MiniCpm4Config::from_config_str("{}").is_err());
        assert!(MiniCpm4Config::from_config_str("not json").is_err());
    }

    #[test]
    fn residual_lm_from_config_str_applies_the_three_overrides() {
        let body =
            residual_config_json("\"residual_lm_num_layers\": 8, \"residual_lm_no_rope\": true");
        let cfg = MiniCpm4Config::residual_lm_from_config_str(&body).expect("parse");
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.vocab_size, 0);
        assert!(cfg.no_rope);
    }

    #[test]
    fn residual_lm_from_config_str_rejects_missing_keys() {
        let no_rope_missing = residual_config_json("\"residual_lm_num_layers\": 8");
        let err = MiniCpm4Config::residual_lm_from_config_str(&no_rope_missing).unwrap_err();
        assert!(err.to_string().contains("residual_lm_no_rope"), "got {err}");

        let layers_missing = residual_config_json("\"residual_lm_no_rope\": true");
        let err = MiniCpm4Config::residual_lm_from_config_str(&layers_missing).unwrap_err();
        assert!(
            err.to_string().contains("residual_lm_num_layers"),
            "got {err}"
        );
    }
}
