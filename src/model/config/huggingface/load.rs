//! Config file loading: try boostr's own formats, then fall back to
//! HuggingFace's `config.json`.

use super::types::HuggingFaceConfig;
use crate::error::{Error, Result};
use crate::model::config::universal::UniversalConfig;
use std::path::Path;

/// Load config, attempting both UniversalConfig and HuggingFace formats
pub fn load_config_auto<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    // Try UniversalConfig first (our native format)
    if let Ok(config) = serde_json::from_str::<UniversalConfig>(&content)
        && config.validate().is_ok()
    {
        return Ok(config);
    }

    // Try YAML format
    if let Ok(config) = serde_saphyr::from_str::<UniversalConfig>(&content)
        && config.validate().is_ok()
    {
        return Ok(config);
    }

    // Try HuggingFace format
    if let Ok(hf_config) = HuggingFaceConfig::from_json(&content) {
        let config = hf_config.to_universal();
        config.validate()?;
        return Ok(config);
    }

    Err(Error::ModelError {
        reason: "Failed to parse config as UniversalConfig, YAML, or HuggingFace format".into(),
    })
}

/// Load HuggingFace config.json and convert to UniversalConfig
pub fn load_huggingface_config<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let hf_config = HuggingFaceConfig::load(path)?;
    let config = hf_config.to_universal();
    config.validate()?;
    Ok(config)
}
