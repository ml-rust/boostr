//! Where a VoxCPM2 transformer stack's weights come from.
//!
//! A pure source selector: it names files and finds the tokenizer beside
//! them, and every loader entry point on `VoxCpm2Model` takes one. The
//! engine that turns it into a running clone pipeline lives in
//! `boostr-audio`.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Where the transformer stack's weights come from. The AudioVAE is always a
/// separate file.
#[derive(Debug, Clone)]
pub enum VoxCpm2Weights {
    /// A checkpoint directory: `config.json`, `model.safetensors`,
    /// `tokenizer.json`.
    Checkpoint(PathBuf),
    /// A single GGUF written by compressr. Carries no `config.json` and no
    /// tokenizer, so `config` points at the checkpoint's.
    Gguf { path: PathBuf, config: PathBuf },
    /// A single TCF written by compressr, on the same terms as GGUF.
    Tcf { path: PathBuf, config: PathBuf },
}

impl VoxCpm2Weights {
    /// `tokenizer.json` for these weights: inside a checkpoint, or beside a
    /// single-file model, or beside its config.
    pub fn tokenizer_path(&self) -> Result<PathBuf> {
        match self {
            Self::Checkpoint(dir) => Ok(dir.join("tokenizer.json")),
            Self::Gguf { path, config } | Self::Tcf { path, config } => {
                let beside = |p: &Path| {
                    p.parent()
                        .map(|dir| dir.join("tokenizer.json"))
                        .filter(|candidate| candidate.is_file())
                };
                beside(path)
                    .or_else(|| beside(config))
                    .ok_or_else(|| Error::ModelError {
                        reason: format!(
                            "no tokenizer.json beside {} or beside {}: a single-file model \
                             carries none",
                            path.display(),
                            config.display()
                        ),
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_path_for_a_single_file_model_needs_a_neighbour() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_tok");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let weights = VoxCpm2Weights::Gguf {
            path: dir.join("m.gguf"),
            config: dir.join("cfg").join("config.json"),
        };
        assert!(weights.tokenizer_path().is_err());
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(
            weights.tokenizer_path().unwrap(),
            dir.join("tokenizer.json")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
