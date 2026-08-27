//! Configuration for VoxCPM2's `fsq_layer` (finite scalar quantization
//! bottleneck) and the six auxiliary projections, resolved from the
//! checkpoint's `config.json`.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// Resolved config for [`crate::model::audio::voxcpm::fsq::ScalarQuantization`]
/// and [`crate::model::audio::voxcpm::fsq::AuxProjections`].
///
/// `latent_dim`/`scale` come from the checkpoint's TOP-level `config.json`
/// (`scalar_quantization_latent_dim`, `scalar_quantization_scale`) — read via
/// [`Self::from_config_json`], never hardcoded. `lm_hidden`/`dit_hidden` are
/// architecture constants verified against the checkpoint (`base_lm`'s
/// `hidden_size` and `feat_decoder`'s `hidden_dim`, matching
/// [`crate::model::audio::voxcpm::minicpm4::config::MiniCpm4Config`]'s
/// `hidden_size` and
/// [`crate::model::audio::voxcpm::local_dit::LocalDitConfig`]'s `hidden_dim`),
/// not present under a `scalar_quantization_*` key.
#[derive(Debug, Clone, Copy)]
pub struct FsqConfig {
    /// `fsq_layer` bottleneck width — top-level `scalar_quantization_latent_dim`
    /// (512).
    pub latent_dim: usize,
    /// Rounding-grid divisor — top-level `scalar_quantization_scale` (9).
    pub scale: f32,
    /// `base_lm`'s hidden width (2048). `fsq_layer.in_proj` reads it,
    /// `fsq_layer.out_proj`/`enc_to_lm_proj`/`stop_proj`/`stop_head` write or
    /// read it, and `fusion_concat_proj`'s input is `2 * lm_hidden`.
    pub lm_hidden: usize,
    /// `feat_decoder`'s hidden width (1024). `lm_to_dit_proj`/
    /// `res_to_dit_proj` write it.
    pub dit_hidden: usize,
}

impl Default for FsqConfig {
    /// Architecture constants verified against the VoxCPM2 checkpoint.
    ///
    /// `latent_dim`/`scale` here are the checkpoint's real values, but use
    /// [`FsqConfig::from_config_json`] to read them from a specific
    /// checkpoint rather than relying on this default.
    fn default() -> Self {
        Self {
            latent_dim: 512,
            scale: 9.0,
            lm_hidden: 2048,
            dit_hidden: 1024,
        }
    }
}

impl FsqConfig {
    /// Parse `scalar_quantization_latent_dim`/`scalar_quantization_scale` out
    /// of a VoxCPM2 `config.json`'s TOP level (not a sub-object).
    /// `lm_hidden`/`dit_hidden` are filled from [`FsqConfig::default`] — see
    /// that impl's docs for why they are not read from JSON.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.as_ref().display()),
        })?;
        Self::from_config_str(&content)
    }

    /// Parse the same top-level keys out of the VERBATIM CONTENTS of a
    /// `config.json`. Split from [`from_config_json`](Self::from_config_json)
    /// so a GGUF's `voxcpm2.config_json` metadata key runs through exactly
    /// this parse.
    pub fn from_config_str(content: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(content).map_err(|e| Error::ModelError {
            reason: format!("invalid VoxCPM2 config.json: {e}"),
        })?;
        let default = FsqConfig::default();
        Ok(FsqConfig {
            latent_dim: raw.scalar_quantization_latent_dim,
            scale: raw.scalar_quantization_scale,
            lm_hidden: default.lm_hidden,
            dit_hidden: default.dit_hidden,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    scalar_quantization_latent_dim: usize,
    scalar_quantization_scale: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_verified_checkpoint_values() {
        let cfg = FsqConfig::default();
        assert_eq!(cfg.latent_dim, 512);
        assert_eq!(cfg.scale, 9.0);
        assert_eq!(cfg.lm_hidden, 2048);
        assert_eq!(cfg.dit_hidden, 1024);
    }

    #[test]
    fn rejects_missing_file() {
        assert!(FsqConfig::from_config_json("/nonexistent/config.json").is_err());
    }
}
