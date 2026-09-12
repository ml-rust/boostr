//! Checkpoint file names and the six-config bundle the transformer stack
//! resolves from ONE `config.json`.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::fsq::FsqConfig;
use crate::model::audio::voxcpm::local_dit::LocalDitConfig;
use crate::model::audio::voxcpm::local_encoder::LocalEncoderConfig;
use crate::model::audio::voxcpm::minicpm4::MiniCpm4Config;
use crate::model::audio::voxcpm::model::config::VoxCpm2Config;
use std::path::Path;

/// Checkpoint file name holding the transformer stack.
pub const DEFAULT_WEIGHTS_FILE: &str = "model.safetensors";
/// Checkpoint file name holding the architecture config.
pub const DEFAULT_CONFIG_FILE: &str = "config.json";

/// The six configs the transformer stack needs, all resolved from ONE
/// `config.json`.
///
/// Grouped so the file entry point and the GGUF entry point share both the
/// parse and the sub-model walk: a GGUF holds the same architecture under the
/// same tensor names, and only the byte container differs.
pub(crate) struct StackConfigs {
    pub(crate) model: VoxCpm2Config,
    pub(crate) base_lm: MiniCpm4Config,
    pub(crate) residual_lm: MiniCpm4Config,
    pub(crate) encoder: LocalEncoderConfig,
    pub(crate) dit: LocalDitConfig,
    pub(crate) fsq: FsqConfig,
}

impl StackConfigs {
    /// Resolve all six out of the verbatim contents of a `config.json`.
    pub(crate) fn from_config_str(content: &str) -> Result<Self> {
        Ok(Self {
            model: VoxCpm2Config::from_config_str(content)?,
            base_lm: MiniCpm4Config::from_config_str(content)?,
            residual_lm: MiniCpm4Config::residual_lm_from_config_str(content)?,
            encoder: LocalEncoderConfig::from_config_str(content)?,
            dit: LocalDitConfig::from_config_str(content)?,
            fsq: FsqConfig::from_config_str(content)?,
        })
    }

    /// Read a `config.json` once and resolve all six out of it. The six
    /// per-type `from_config_json` constructors each read the file
    /// themselves, which would be six reads of the same bytes here.
    pub(crate) fn from_config_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.display()),
        })?;
        Self::from_config_str(&content)
    }
}
