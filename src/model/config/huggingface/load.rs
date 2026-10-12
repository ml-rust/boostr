//! Config file loading: try boostr's own formats, then fall back to
//! HuggingFace's `config.json`.

use super::types::HuggingFaceConfig;
use crate::error::{Error, Result};
use crate::model::config::universal::UniversalConfig;
use crate::model::qwen35::{is_qwen35_hf_config, qwen35_config_from_hf};
use std::path::Path;

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })
}

/// The `qwen35` config when `content` is a Hugging Face Qwen3.5
/// `config.json`, else `None`. Qwen3.5 nests `vocab_size` and `hidden_size`
/// under `text_config`, which the generic [`HuggingFaceConfig`] parse cannot
/// read.
fn qwen35_config(content: &str) -> Option<Result<UniversalConfig>> {
    let json = serde_json::from_str::<serde_json::Value>(content).ok()?;
    is_qwen35_hf_config(&json).then(|| qwen35_config_from_hf(&json))
}

/// Load config, attempting both UniversalConfig and HuggingFace formats
pub fn load_config_auto<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let content = read(path.as_ref())?;

    if let Some(config) = qwen35_config(&content) {
        return config;
    }

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
        let config = hf_config.to_universal()?;
        config.validate()?;
        return Ok(config);
    }

    Err(Error::ModelError {
        reason: "Failed to parse config as UniversalConfig, YAML, or HuggingFace format".into(),
    })
}

/// Load HuggingFace config.json and convert to UniversalConfig
pub fn load_huggingface_config<P: AsRef<Path>>(path: P) -> Result<UniversalConfig> {
    let content = read(path.as_ref())?;
    if let Some(config) = qwen35_config(&content) {
        return config;
    }
    let hf_config = HuggingFaceConfig::from_json(&content)?;
    let config = hf_config.to_universal()?;
    config.validate()?;
    Ok(config)
}
