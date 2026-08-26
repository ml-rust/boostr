//! Configuration for VoxCPM2's MiniCPM4 decoder-only transformer, resolved
//! from the checkpoint's `config.json`.
//!
//! Every architectural knob is config-driven on purpose: VoxCPM2's
//! `residual_lm` is the SAME architecture as `base_lm` with a different
//! config (8 layers, `vocab_size` 0, hence no `embed_tokens` table), so it
//! becomes a second [`MiniCpm4Config`] rather than a forked module.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// `config.json` sub-object holding `base_lm`'s architecture.
pub const DEFAULT_CONFIG_SECTION: &str = "lm_config";

/// Resolved config for
/// [`MiniCpm4Model`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Model).
///
/// Read from a single `config.json` sub-object (`lm_config` for `base_lm`)
/// by [`MiniCpm4Config::from_config_json_section`].
#[derive(Debug, Clone)]
pub struct MiniCpm4Config {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    /// Per-head width, read from the checkpoint's `kv_channels` — NEVER
    /// derived as `hidden_size / num_heads`. The two agree on `base_lm`
    /// (2048/16 == 128 == `kv_channels`), which is exactly what makes the
    /// derivation a latent bug: the local-encoder sibling has `hidden_dim`
    /// 1024 with `kv_channels` 128, where deriving gives 64 and silently
    /// mis-shapes every projection.
    pub head_dim: usize,
    /// Rows in the `embed_tokens` table. `0` means the checkpoint carries no
    /// embedding table at all (VoxCPM2's `residual_lm`), and the loader then
    /// builds a model whose
    /// [`embed`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::embed)
    /// errors instead of silently returning zeros.
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    /// Per-dimension LongRoPE short-context rescale, length `head_dim / 2`.
    pub rope_short_factor: Vec<f32>,
    /// Per-dimension LongRoPE long-context rescale, length `head_dim / 2`.
    /// `RoPE::precompute_freqs` selects this over `rope_short_factor` only
    /// when `max_position_embeddings > original_max_position_embeddings`;
    /// on this checkpoint the two are equal (32768 == 32768), so
    /// `rope_short_factor` is always selected in practice and the LongRoPE
    /// `attention_scaling` collapses to 1.0.
    pub rope_long_factor: Vec<f32>,
}

impl Default for MiniCpm4Config {
    /// `base_lm`'s architecture constants, verified against the VoxCPM2
    /// checkpoint (254 tensors: 1 embedding + 28 x 9 per-layer + 1 final
    /// norm).
    ///
    /// `rope_short_factor`/`rope_long_factor` are the RoPE IDENTITY
    /// (all-ones) here, NOT the checkpoint's real per-dimension values —
    /// those must come from the checkpoint's `config.json` via
    /// [`MiniCpm4Config::from_config_json`]. Using this `Default` as-is
    /// silently applies unscaled RoPE, which is numerically wrong for this
    /// checkpoint.
    fn default() -> Self {
        let head_dim = 128;
        Self {
            num_layers: 28,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim,
            vocab_size: 73448,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 32768,
            original_max_position_embeddings: 32768,
            rope_short_factor: vec![1.0; head_dim / 2],
            rope_long_factor: vec![1.0; head_dim / 2],
        }
    }
}

impl MiniCpm4Config {
    /// Parse `lm_config` out of a VoxCPM2 `config.json` — i.e. `base_lm`.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_config_json_section(path, DEFAULT_CONFIG_SECTION)
    }

    /// Parse an arbitrary sub-object of a VoxCPM2 `config.json`.
    ///
    /// `section` is a top-level key (`"lm_config"` for `base_lm`). The
    /// section is taken as a parameter because `residual_lm` is the same
    /// architecture under a different key, and must not require a second
    /// parser.
    pub fn from_config_json_section<P: AsRef<Path>>(path: P, section: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.as_ref().display()),
        })?;
        let root: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| Error::ModelError {
                reason: format!("invalid VoxCPM2 config.json: {e}"),
            })?;
        let sub = root.get(section).ok_or_else(|| Error::ModelError {
            reason: format!("VoxCPM2 config.json has no `{section}` object"),
        })?;
        let raw: RawLmConfig =
            serde_json::from_value(sub.clone()).map_err(|e| Error::ModelError {
                reason: format!("invalid VoxCPM2 config.json `{section}`: {e}"),
            })?;
        raw.resolve()
    }

    /// Whether this instantiation owns an `embed_tokens` table.
    ///
    /// `false` for a `vocab_size == 0` config (`residual_lm`), which is fed
    /// pre-computed embeddings only.
    pub fn has_embedding(&self) -> bool {
        self.vocab_size > 0
    }
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
        })
    }
}

#[cfg(test)]
mod tests {
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
}
