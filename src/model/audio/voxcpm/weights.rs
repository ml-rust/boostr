//! Where a VoxCPM2 transformer stack's weights come from.
//!
//! A pure source selector: it names files and finds the tokenizer inside or
//! beside them, and every loader entry point on `VoxCpm2Model` takes one.
//! The engine that turns it into a running clone pipeline lives in
//! `boostr-audio`.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
use crate::model::audio::voxcpm::gguf_keys::GGUF_TOKENIZER_JSON_KEY;

/// Where the transformer stack's weights come from.
///
/// The AudioVAE rides along only in a single-file model: compressr embeds
/// it (folded, dense, under `vae.`) when the directory it converts holds
/// `audiovae.pth`. A loader reads the embedded copy when present and the
/// separate `audiovae.pth`/`audiovae.safetensors` otherwise; the embedded
/// copy wins when both are offered. A checkpoint directory never embeds one.
#[derive(Debug, Clone)]
pub enum VoxCpm2Weights {
    /// A checkpoint directory: `config.json`, `model.safetensors`,
    /// `tokenizer.json`. The AudioVAE is a separate file.
    Checkpoint(PathBuf),
    /// A single GGUF written by compressr. One that embeds its
    /// `config.json` (as `voxcpm2.config_json`) and `tokenizer.json` (as
    /// `tokenizer.huggingface.json`) needs nothing beside it, and `config`
    /// is `None`. An older or third-party GGUF carries neither: `config`
    /// names the checkpoint's `config.json`, and `tokenizer.json` is looked
    /// for beside the file and then beside `config`. The embedded copy wins
    /// when both are offered.
    Gguf {
        path: PathBuf,
        config: Option<PathBuf>,
    },
    /// A single TCF written by compressr, on the same terms as GGUF except
    /// that a TCF has no metadata map, so `config` is always read and the
    /// tokenizer is always a file beside the TCF or beside `config`.
    Tcf { path: PathBuf, config: PathBuf },
}

/// Where the tokenizer's `tokenizer.json` bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    /// The file's text, read out of a GGUF's metadata.
    Embedded(String),
    /// A `tokenizer.json` on disk.
    File(PathBuf),
}

impl VoxCpm2Weights {
    /// The tokenizer for these weights: inside a checkpoint directory,
    /// embedded in a GGUF, or a `tokenizer.json` beside a single-file model
    /// or beside its config.
    ///
    /// # Errors
    /// A GGUF that cannot be opened, or a single-file model that embeds no
    /// tokenizer and has no `tokenizer.json` beside it or beside its config.
    pub fn tokenizer_source(&self) -> Result<TokenizerSource> {
        match self {
            Self::Checkpoint(dir) => Ok(TokenizerSource::File(dir.join("tokenizer.json"))),
            Self::Gguf { path, config } => {
                let gguf = Gguf::open(path)?;
                if let Some(text) = gguf.metadata().get_string(GGUF_TOKENIZER_JSON_KEY) {
                    return Ok(TokenizerSource::Embedded(text.to_string()));
                }
                tokenizer_file_beside(path, config.as_deref())
            }
            Self::Tcf { path, config } => tokenizer_file_beside(path, Some(config)),
        }
    }
}

/// `tokenizer.json` beside `path`, else beside `config`, else an error
/// naming both places.
fn tokenizer_file_beside(path: &Path, config: Option<&Path>) -> Result<TokenizerSource> {
    let beside = |p: &Path| {
        p.parent()
            .map(|dir| dir.join("tokenizer.json"))
            .filter(|candidate| candidate.is_file())
    };
    beside(path)
        .or_else(|| config.and_then(beside))
        .map(TokenizerSource::File)
        .ok_or_else(|| {
            let beside_config = config
                .map(|c| format!(" or beside {}", c.display()))
                .unwrap_or_default();
            Error::ModelError {
                reason: format!(
                    "no tokenizer.json embedded in or beside {}{beside_config}: convert with \
                     compressr from a directory holding tokenizer.json, or put the file beside \
                     the model",
                    path.display(),
                ),
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF v3 image with no tensors and the given string metadata.
    fn gguf_bytes(metadata: &[(&str, &str)]) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            put_str(&mut out, key);
            // Value type 8 is `String`.
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(&mut out, value);
        }
        let aligned = out.len().div_ceil(32) * 32;
        out.resize(aligned, 0);
        out
    }

    fn fresh_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_gguf_with_the_key_yields_the_embedded_text() {
        let dir = fresh_dir("boostr_voxcpm2_weights_embedded_tok");
        let path = dir.join("m.gguf");
        let text = "{\"model\": {\"type\": \"BPE\"}}";
        std::fs::write(&path, gguf_bytes(&[(GGUF_TOKENIZER_JSON_KEY, text)])).unwrap();
        // A file beside it does not win over the embedded copy.
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        let weights = VoxCpm2Weights::Gguf { path, config: None };
        assert_eq!(
            weights.tokenizer_source().unwrap(),
            TokenizerSource::Embedded(text.to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_gguf_without_the_key_falls_back_to_a_neighbour() {
        let dir = fresh_dir("boostr_voxcpm2_weights_file_tok");
        let path = dir.join("m.gguf");
        std::fs::write(&path, gguf_bytes(&[("general.architecture", "voxcpm2")])).unwrap();
        let weights = VoxCpm2Weights::Gguf {
            path,
            config: Some(dir.join("cfg").join("config.json")),
        };
        assert!(weights.tokenizer_source().is_err());
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(
            weights.tokenizer_source().unwrap(),
            TokenizerSource::File(dir.join("tokenizer.json"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_gguf_without_the_key_falls_back_to_the_config_neighbour() {
        let dir = fresh_dir("boostr_voxcpm2_weights_config_tok");
        let cfg_dir = dir.join("cfg");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::write(cfg_dir.join("tokenizer.json"), b"{}").unwrap();
        let path = dir.join("m.gguf");
        std::fs::write(&path, gguf_bytes(&[])).unwrap();
        let weights = VoxCpm2Weights::Gguf {
            path,
            config: Some(cfg_dir.join("config.json")),
        };
        assert_eq!(
            weights.tokenizer_source().unwrap(),
            TokenizerSource::File(cfg_dir.join("tokenizer.json"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tcf_needs_a_neighbour() {
        let dir = fresh_dir("boostr_voxcpm2_weights_tcf_tok");
        let weights = VoxCpm2Weights::Tcf {
            path: dir.join("m.tcf"),
            config: dir.join("cfg").join("config.json"),
        };
        assert!(weights.tokenizer_source().is_err());
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(
            weights.tokenizer_source().unwrap(),
            TokenizerSource::File(dir.join("tokenizer.json"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_checkpoint_names_its_own_file() {
        let weights = VoxCpm2Weights::Checkpoint(PathBuf::from("/ckpt"));
        assert_eq!(
            weights.tokenizer_source().unwrap(),
            TokenizerSource::File(PathBuf::from("/ckpt/tokenizer.json"))
        );
    }
}
