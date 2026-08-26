//! Config constants for the VoxCPM2 end-to-end orchestrator: the patch
//! geometry the reference-audio prefix is built from, and the three special
//! token ids that delimit it.
//!
//! Everything here is read from the checkpoint's TOP-level `config.json`
//! (`patch_size`, `feat_dim`), never hardcoded — see
//! [`VoxCpm2Config::from_config_json`]. The special token ids are the
//! reference implementation's own literals and have no `config.json` key.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::vae::encoder::HOP_LENGTH;
use serde::Deserialize;
use std::path::Path;

/// Marks the start of the reference-audio span. Its own position carries a
/// ZERO audio patch and is a TEXT position (`text_mask` 1, `audio_mask` 0).
pub const REF_AUDIO_START_ID: u32 = 103;

/// Marks the end of the reference-audio span. Also a TEXT position, also
/// backed by a zero audio patch.
pub const REF_AUDIO_END_ID: u32 = 104;

/// Terminates the text prompt. `text_token_ids` handed to
/// [`prefill`](crate::model::audio::voxcpm::model::VoxCpm2Model::prefill)
/// must already end with this id — boostr does not tokenize, so it cannot
/// append it.
pub const AUDIO_START_ID: u32 = 101;

/// Filler token id written at every reference-audio position. The value is
/// irrelevant to the model: those positions carry `text_mask == 0`, so their
/// embedding is multiplied out before it reaches `base_lm`.
pub const REF_AUDIO_FILLER_ID: u32 = 0;

/// Patch geometry for the VoxCPM2 orchestrator.
#[derive(Debug, Clone, Copy)]
pub struct VoxCpm2Config {
    /// Audio-VAE frames folded into one LM position — top-level
    /// `patch_size` (4).
    pub patch_size: usize,
    /// Per-frame latent width — top-level `feat_dim` (64). This is the
    /// AudioVAE encoder's channel count, and `feat_encoder`'s `in_proj`
    /// input width.
    pub feat_dim: usize,
}

impl Default for VoxCpm2Config {
    /// Architecture constants verified against the VoxCPM2 checkpoint. Use
    /// [`VoxCpm2Config::from_config_json`] to read them from a specific
    /// checkpoint rather than relying on this default.
    fn default() -> Self {
        Self {
            patch_size: 4,
            feat_dim: 64,
        }
    }
}

impl VoxCpm2Config {
    /// Samples of 16 kHz reference audio the wav must be right-padded to a
    /// multiple of BEFORE the VAE encode: `patch_size * 640` (2560).
    ///
    /// This is NOT the AudioVAE's own modulus. `AudioVaeEncoder::forward`
    /// already right-pads to a multiple of [`HOP_LENGTH`] (640) on its own,
    /// which only guarantees a whole number of latent FRAMES. The patch fold
    /// needs a whole number of `patch_size`-frame PATCHES, i.e. four times
    /// that. Relying on the VAE's padding alone leaves a frame count that is
    /// not divisible by `patch_size` and the fold then fails (or, with a
    /// truncating fold, silently drops the tail of the reference).
    pub fn ref_pad_multiple(&self) -> usize {
        self.patch_size * HOP_LENGTH
    }

    /// Parse top-level `patch_size` and `feat_dim` out of a VoxCPM2
    /// `config.json`.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.as_ref().display()),
        })?;
        let raw: RawConfig = serde_json::from_str(&content).map_err(|e| Error::ModelError {
            reason: format!("invalid VoxCPM2 config.json: {e}"),
        })?;
        if raw.patch_size == 0 {
            return Err(Error::ModelError {
                reason: "VoxCPM2 config.json has patch_size 0".to_string(),
            });
        }
        if raw.feat_dim == 0 {
            return Err(Error::ModelError {
                reason: "VoxCPM2 config.json has feat_dim 0".to_string(),
            });
        }
        Ok(Self {
            patch_size: raw.patch_size,
            feat_dim: raw.feat_dim,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    patch_size: usize,
    feat_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_pad_multiple_is_patch_size_times_vae_hop() {
        let cfg = VoxCpm2Config::default();
        assert_eq!(cfg.ref_pad_multiple(), 2560);
        // The trap this constant guards: the VAE's own modulus is 640, and
        // 640 is NOT enough.
        assert_ne!(cfg.ref_pad_multiple(), HOP_LENGTH);
    }

    #[test]
    fn ref_pad_multiple_tracks_patch_size() {
        let cfg = VoxCpm2Config {
            patch_size: 8,
            feat_dim: 64,
        };
        assert_eq!(cfg.ref_pad_multiple(), 8 * HOP_LENGTH);
    }

    #[test]
    fn rejects_missing_file() {
        assert!(VoxCpm2Config::from_config_json("/nonexistent/config.json").is_err());
    }
}
