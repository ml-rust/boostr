//! Configuration for VoxCPM2's `feat_encoder` local encoder ("locenc"),
//! resolved from the checkpoint's `config.json`.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// Resolved config for [`crate::model::audio::voxcpm::local_encoder::LocalEncoder`].
///
/// The checkpoint's `encoder_config` overrides `hidden_dim`/`ffn_dim`/
/// `num_heads`/`num_layers`/`kv_channels` (head_dim) on top of `lm_config`'s
/// `rope_theta`/`rope_scaling`/`rms_norm_eps`/`num_key_value_heads`/
/// `max_position_embeddings` — `num_key_value_heads` (2) is inherited from
/// `lm_config`, not derivable from `encoder_config` alone.
#[derive(Debug, Clone)]
pub struct LocalEncoderConfig {
    /// `in_proj` input width (per-patch feature dim).
    pub patch_dim: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    /// Sequence length per `(batch, frame)`: 1 CLS + `patch_dim`'s patch count.
    pub num_positions: usize,
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
    /// `rope_short_factor` is always selected in practice.
    pub rope_long_factor: Vec<f32>,
}

impl Default for LocalEncoderConfig {
    /// Architecture constants verified against the VoxCPM2 checkpoint.
    ///
    /// `rope_short_factor`/`rope_long_factor` are set to the RoPE IDENTITY
    /// (all-ones) here, NOT the checkpoint's real per-dimension values
    /// (~0.998-1.03) — those must come from the checkpoint's `config.json`
    /// via [`LocalEncoderConfig::from_config_json`]. Using this `Default`
    /// as-is silently applies unscaled RoPE, which is numerically wrong for
    /// this checkpoint.
    fn default() -> Self {
        let head_dim = 128;
        Self {
            patch_dim: 64,
            hidden_dim: 1024,
            ffn_dim: 4096,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim,
            num_layers: 12,
            num_positions: 5,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 32768,
            original_max_position_embeddings: 32768,
            rope_short_factor: vec![1.0; head_dim / 2],
            rope_long_factor: vec![1.0; head_dim / 2],
        }
    }
}

impl LocalEncoderConfig {
    /// Parse `lm_config`/`encoder_config` out of a VoxCPM2 `config.json`,
    /// applying `encoder_config`'s overrides on top of `lm_config`'s
    /// RoPE/norm settings. `patch_dim` and `num_positions` are not present
    /// in the checkpoint's config (they come from the upstream patch-folding
    /// stage), so [`Default::default`] supplies them.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.as_ref().display()),
        })?;
        let raw: RawConfig = serde_json::from_str(&content).map_err(|e| Error::ModelError {
            reason: format!("invalid VoxCPM2 config.json: {e}"),
        })?;
        raw.resolve()
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    lm_config: RawLmConfig,
    encoder_config: RawEncoderConfig,
}

#[derive(Debug, Deserialize)]
struct RawLmConfig {
    rope_theta: f32,
    rope_scaling: RawRopeScaling,
    rms_norm_eps: f32,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
}

#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawEncoderConfig {
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_layers: usize,
    kv_channels: usize,
}

impl RawConfig {
    fn resolve(self) -> Result<LocalEncoderConfig> {
        let half_dim = self.encoder_config.kv_channels / 2;
        if self.lm_config.rope_scaling.short_factor.len() != half_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_scaling.short_factor has {} entries, expected {half_dim} \
                     (kv_channels/2)",
                    self.lm_config.rope_scaling.short_factor.len()
                ),
            });
        }
        let default = LocalEncoderConfig::default();
        Ok(LocalEncoderConfig {
            patch_dim: default.patch_dim,
            hidden_dim: self.encoder_config.hidden_dim,
            ffn_dim: self.encoder_config.ffn_dim,
            num_heads: self.encoder_config.num_heads,
            num_kv_heads: self.lm_config.num_key_value_heads,
            head_dim: self.encoder_config.kv_channels,
            num_layers: self.encoder_config.num_layers,
            num_positions: default.num_positions,
            rms_norm_eps: self.lm_config.rms_norm_eps,
            rope_theta: self.lm_config.rope_theta,
            max_position_embeddings: self.lm_config.max_position_embeddings,
            original_max_position_embeddings: self
                .lm_config
                .rope_scaling
                .original_max_position_embeddings
                .unwrap_or(self.lm_config.max_position_embeddings),
            rope_short_factor: self.lm_config.rope_scaling.short_factor,
            rope_long_factor: self.lm_config.rope_scaling.long_factor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_head_dim_matches_short_factor_len() {
        let cfg = LocalEncoderConfig::default();
        assert_eq!(cfg.rope_short_factor.len(), cfg.head_dim / 2);
    }

    #[test]
    fn rejects_missing_file() {
        assert!(LocalEncoderConfig::from_config_json("/nonexistent/config.json").is_err());
    }
}
