//! Configuration for VoxCPM2's `feat_decoder` local DiT ("locdit"),
//! resolved from the checkpoint's `config.json`.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// Resolved config for
/// [`crate::model::audio::voxcpm::local_dit::LocalDit`].
///
/// The checkpoint's `dit_config` supplies the DiT's own hidden/ffn/heads/layers
/// sizes; several other REQUIRED fields are absent from `dit_config` and come
/// from `lm_config` instead — `num_key_value_heads` (2), `kv_channels` (the
/// DiT's real `head_dim`, 128), `rms_norm_eps`, `rope_theta`, `rope_scaling`
/// (longrope short/long factors + `original_max_position_embeddings`), and
/// `max_position_embeddings`. `feat_dim` (64) and `patch_size` (4) come from
/// the checkpoint's TOP level, not `dit_config`.
#[derive(Debug, Clone)]
pub struct LocalDitConfig {
    /// `in_proj`/`cond_proj`/`out_proj` width (64) — top-level `feat_dim`.
    pub feat_dim: usize,
    /// Top-level `patch_size` (4). Sizes the `cond`/`x` spans of the
    /// assembled sequence — see [`Self::sequence_len`].
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    /// The DiT's real attention head dim. `dit_config.kv_channels` is `None`
    /// on this checkpoint — this is `lm_config.kv_channels` (128), NOT
    /// `hidden_dim / num_heads` (which would be 1024/16 = 64, wrong).
    pub head_dim: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    /// Per-dimension LongRoPE short-context rescale, length `head_dim / 2`.
    pub rope_short_factor: Vec<f32>,
    /// Per-dimension LongRoPE long-context rescale, length `head_dim / 2`.
    pub rope_long_factor: Vec<f32>,
    /// `dit_config.mean_mode` from the checkpoint's JSON is a DEAD key: the
    /// reference's pydantic field on `VoxCPMDitConfig` is named
    /// `dit_mean_mode`, so pydantic silently drops the unmatched `mean_mode`
    /// JSON key and `dit_mean_mode` keeps its default `false`
    /// (`voxcpm/model/voxcpm2.py`: `VoxCPMDitConfig.dit_mean_mode: bool =
    /// False`, then `mean_mode=config.dit_config.dit_mean_mode`). This field
    /// is therefore always `false` for this checkpoint; it is NOT read from
    /// `dit_config.mean_mode`.
    pub mean_mode: bool,
}

impl Default for LocalDitConfig {
    /// Architecture constants verified against the VoxCPM2 checkpoint.
    ///
    /// As with [`crate::model::audio::voxcpm::local_encoder::LocalEncoderConfig`],
    /// `rope_short_factor`/`rope_long_factor` are the RoPE IDENTITY (all-ones)
    /// here, NOT the checkpoint's real per-dimension values — those must come
    /// from [`LocalDitConfig::from_config_json`].
    fn default() -> Self {
        let head_dim = 128;
        Self {
            feat_dim: 64,
            patch_size: 4,
            hidden_dim: 1024,
            ffn_dim: 4096,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim,
            num_layers: 12,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 32768,
            original_max_position_embeddings: 32768,
            rope_short_factor: vec![1.0; head_dim / 2],
            rope_long_factor: vec![1.0; head_dim / 2],
            mean_mode: false,
        }
    }
}

impl LocalDitConfig {
    /// Parse `lm_config`/`dit_config` out of a VoxCPM2 `config.json`. See
    /// the field docs above for which JSON section each value comes from.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.as_ref().display()),
        })?;
        let raw: RawConfig = serde_json::from_str(&content).map_err(|e| Error::ModelError {
            reason: format!("invalid VoxCPM2 config.json: {e}"),
        })?;
        raw.resolve()
    }

    /// The DiT's assembled sequence length: `mu` (2 tokens) + `t` (1) +
    /// `cond` (`patch_size`) + `x` (`patch_size`). Derived, never hardcoded —
    /// `2 + 1 + patch_size + patch_size` (11 for `patch_size = 4`).
    pub fn sequence_len(&self) -> usize {
        2 + 1 + self.patch_size + self.patch_size
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    lm_config: RawLmConfig,
    dit_config: RawDitConfig,
    feat_dim: usize,
    patch_size: usize,
}

#[derive(Debug, Deserialize)]
struct RawLmConfig {
    rope_theta: f32,
    rope_scaling: RawRopeScaling,
    rms_norm_eps: f32,
    num_key_value_heads: usize,
    kv_channels: usize,
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
struct RawDitConfig {
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_layers: usize,
    // Deliberately NOT read: this checkpoint's `dit_config.kv_channels` is
    // JSON `null` and, even when present, is not what the reference uses as
    // `head_dim` for the DiT (`lm_config.kv_channels` is). Kept undeserialized
    // so a `null` value here never fails parsing.
    //
    // `mean_mode` is deliberately NOT read here either — see
    // `LocalDitConfig::mean_mode`'s doc comment for why it would be a dead
    // read even if a field existed.
}

impl RawConfig {
    fn resolve(self) -> Result<LocalDitConfig> {
        let head_dim = self.lm_config.kv_channels;
        let half_dim = head_dim / 2;
        if self.lm_config.rope_scaling.short_factor.len() != half_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_scaling.short_factor has {} entries, expected {half_dim} \
                     (kv_channels/2)",
                    self.lm_config.rope_scaling.short_factor.len()
                ),
            });
        }
        let default = LocalDitConfig::default();
        Ok(LocalDitConfig {
            feat_dim: self.feat_dim,
            patch_size: self.patch_size,
            hidden_dim: self.dit_config.hidden_dim,
            ffn_dim: self.dit_config.ffn_dim,
            num_heads: self.dit_config.num_heads,
            num_kv_heads: self.lm_config.num_key_value_heads,
            head_dim,
            num_layers: self.dit_config.num_layers,
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
            mean_mode: default.mean_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_head_dim_matches_short_factor_len() {
        let cfg = LocalDitConfig::default();
        assert_eq!(cfg.rope_short_factor.len(), cfg.head_dim / 2);
    }

    #[test]
    fn default_head_dim_is_kv_channels_not_hidden_over_heads() {
        let cfg = LocalDitConfig::default();
        assert_eq!(cfg.head_dim, 128);
        assert_ne!(cfg.head_dim, cfg.hidden_dim / cfg.num_heads);
    }

    #[test]
    fn sequence_len_is_derived_from_patch_size() {
        let cfg = LocalDitConfig::default();
        assert_eq!(cfg.sequence_len(), 11);
        let mut cfg2 = cfg.clone();
        cfg2.patch_size = 8;
        assert_eq!(cfg2.sequence_len(), 2 + 1 + 8 + 8);
    }

    #[test]
    fn default_mean_mode_is_false() {
        assert!(!LocalDitConfig::default().mean_mode);
    }

    #[test]
    fn rejects_missing_file() {
        assert!(LocalDitConfig::from_config_json("/nonexistent/config.json").is_err());
    }
}
