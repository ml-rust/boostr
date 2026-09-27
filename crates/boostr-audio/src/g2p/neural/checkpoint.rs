//! Finds the checkpoint file oxidizr wrote into a directory.
//!
//! oxidizr names checkpoints `{prefix}_{name}.safetensors` at
//! `oxidizr/src/trainer.rs:1200` (`save_checkpoint_named`, `name` = `"final"`
//! at the end of training) and `{prefix}_step_{N}.safetensors` for
//! intermediate saves, where `prefix` encodes the model name, sequence
//! length, parameter count and dtype (`checkpoint_prefix`,
//! `oxidizr/src/config.rs:520`) — not reproducible here without duplicating
//! that estimator, so this scans the directory instead of guessing the name.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Picks the checkpoint to load from a directory holding one or more
/// `*.safetensors` files: the one named `*_final.safetensors` if present,
/// else the one with the highest `_step_<N>` suffix, else the sole
/// `*.safetensors` file. Errors, naming every candidate, when the directory
/// holds several files and none of those rules picks one.
pub(crate) fn locate(dir: &Path) -> Result<PathBuf> {
    let mut candidates = candidates_in(dir)?;
    candidates.sort();

    if candidates.is_empty() {
        return Err(Error::ModelError {
            reason: format!("no *.safetensors checkpoint in {}", dir.display()),
        });
    }
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }

    if let Some(final_ckpt) = candidates.iter().find(|p| stem(p).ends_with("_final")) {
        return Ok(final_ckpt.clone());
    }

    let by_step: Vec<(u64, &PathBuf)> = candidates
        .iter()
        .filter_map(|p| step_suffix(&stem(p)).map(|n| (n, p)))
        .collect();
    if by_step.len() == candidates.len()
        && let Some((_, path)) = by_step.iter().max_by_key(|(n, _)| *n)
    {
        return Ok((*path).clone());
    }

    let names: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
    Err(Error::ModelError {
        reason: format!(
            "{} holds {} safetensors files and none is named *_final or *_step_<N>: {}",
            dir.display(),
            candidates.len(),
            names.join(", ")
        ),
    })
}

/// Every `*.safetensors` path directly inside `dir`.
fn candidates_in(dir: &Path) -> Result<Vec<PathBuf>> {
    Ok(std::fs::read_dir(dir)
        .map_err(|e| Error::ModelError {
            reason: format!("reading {}: {e}", dir.display()),
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect())
}

/// The file stem (name without extension), or an empty string.
fn stem(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

/// Parses the trailing `_step_<N>` off a file stem, if present.
fn step_suffix(stem: &str) -> Option<u64> {
    let (_, n) = stem.rsplit_once("_step_")?;
    n.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn picks_the_sole_checkpoint() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        fs::write(
            dir.path().join("g2p-char_128_4m_bf16_step_491.safetensors"),
            b"",
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let picked = locate(dir.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(picked.ends_with("g2p-char_128_4m_bf16_step_491.safetensors"));
    }

    #[test]
    fn prefers_final_over_numbered_steps() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        for name in [
            "g2p-char_128_4m_bf16_step_491.safetensors",
            "g2p-char_128_4m_bf16_final.safetensors",
        ] {
            fs::write(dir.path().join(name), b"").unwrap_or_else(|e| panic!("{e}"));
        }
        let picked = locate(dir.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(picked.ends_with("g2p-char_128_4m_bf16_final.safetensors"));
    }

    #[test]
    fn picks_the_highest_step_when_no_final_exists() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        for name in [
            "g2p-char_128_4m_bf16_step_491.safetensors",
            "g2p-char_128_4m_bf16_step_982.safetensors",
        ] {
            fs::write(dir.path().join(name), b"").unwrap_or_else(|e| panic!("{e}"));
        }
        let picked = locate(dir.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(picked.ends_with("g2p-char_128_4m_bf16_step_982.safetensors"));
    }

    #[test]
    fn ambiguous_directory_names_every_candidate() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        for name in ["a.safetensors", "b.safetensors"] {
            fs::write(dir.path().join(name), b"").unwrap_or_else(|e| panic!("{e}"));
        }
        let err = locate(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("a.safetensors") && msg.contains("b.safetensors"));
    }
}
