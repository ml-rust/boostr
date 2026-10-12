//! PEFT adapter file locations: the config and weights file names, and
//! resolving an adapter directory or weights file to both paths.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// File name of a PEFT adapter's config.
pub const ADAPTER_CONFIG_FILE: &str = "adapter_config.json";
/// File name of a PEFT adapter's SafeTensors weights.
pub const ADAPTER_WEIGHTS_FILE: &str = "adapter_model.safetensors";
pub(super) const ADAPTER_BIN_FILE: &str = "adapter_model.bin";

/// `(config path, weights path)` for an adapter directory or weights file.
pub(super) fn resolve_paths(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let model_error = |reason: String| Error::ModelError { reason };
    let (dir, weights) = if path.is_dir() {
        let weights = path.join(ADAPTER_WEIGHTS_FILE);
        if !weights.is_file() {
            let bin = path.join(ADAPTER_BIN_FILE);
            if bin.is_file() {
                return Err(bin_unsupported(&bin));
            }
            return Err(model_error(format!(
                "PEFT adapter directory {} has no {ADAPTER_WEIGHTS_FILE}",
                path.display()
            )));
        }
        (path.to_path_buf(), weights)
    } else if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => {}
            Some("bin") => return Err(bin_unsupported(path)),
            _ => {
                return Err(model_error(format!(
                    "PEFT adapter file {} is not a .safetensors file",
                    path.display()
                )));
            }
        }
        let dir = path.parent().unwrap_or(Path::new("")).to_path_buf();
        (dir, path.to_path_buf())
    } else {
        return Err(model_error(format!(
            "PEFT adapter path {} does not exist",
            path.display()
        )));
    };
    let config = dir.join(ADAPTER_CONFIG_FILE);
    if !config.is_file() {
        return Err(model_error(format!(
            "PEFT adapter {} has no {ADAPTER_CONFIG_FILE} beside its weights; its rank and alpha \
             are required",
            path.display()
        )));
    }
    Ok((config, weights))
}

fn bin_unsupported(path: &Path) -> Error {
    Error::ModelError {
        reason: format!(
            "PEFT adapter weights {} are in the PyTorch .bin format, which is unsupported; \
             convert them to {ADAPTER_WEIGHTS_FILE}",
            path.display()
        ),
    }
}
