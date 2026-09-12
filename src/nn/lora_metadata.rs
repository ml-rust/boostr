//! LoRA adapter config metadata: rank/alpha/targets carried in a
//! safetensors file's `__metadata__`, so a load can catch a
//! `--rank`/`--targets` mismatch against the model it's applied to instead
//! of failing downstream as an opaque shape error or — worse — silently
//! applying an adapter at the wrong strength. Split out of
//! [`crate::nn::lora_targets`] to keep that file under this crate's
//! soft line-count limit for `nn/*.rs`.

use crate::error::{Error, Result};
use std::collections::HashMap;

/// `__metadata__` key naming the LoRA rank an adapter file was saved at.
pub const LORA_METADATA_RANK_KEY: &str = "lora_rank";
/// `__metadata__` key naming the LoRA alpha an adapter file was saved at.
pub const LORA_METADATA_ALPHA_KEY: &str = "lora_alpha";
/// `__metadata__` key naming the comma-joined LoRA target names an adapter
/// file was saved with.
pub const LORA_METADATA_TARGETS_KEY: &str = "lora_targets";

/// Build the `__metadata__` map
/// [`save_safetensors`](crate::format::safetensors::save_safetensors) writes
/// into a LoRA adapter file: rank, alpha, and target names, so a later load
/// can catch a `--rank`/`--targets` mismatch against the model it is
/// applied to via [`check_lora_metadata`], up front, instead of failing
/// downstream as an opaque shape mismatch or — worse — succeeding with the
/// wrong adapter strength.
pub fn build_lora_metadata(rank: usize, alpha: f32, targets: &[String]) -> HashMap<String, String> {
    let mut meta = HashMap::with_capacity(3);
    meta.insert(LORA_METADATA_RANK_KEY.to_string(), rank.to_string());
    meta.insert(LORA_METADATA_ALPHA_KEY.to_string(), alpha.to_string());
    meta.insert(LORA_METADATA_TARGETS_KEY.to_string(), targets.join(","));
    meta
}

/// Rank, alpha and target names parsed out of an adapter file's
/// `__metadata__` — [`parse_lora_metadata`]'s result, and what
/// `VoxCpm2Model::load_lora_adapter` applies without a caller-supplied
/// `--rank`/`--alpha`/`--targets` to get wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraMetadata {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
}

/// Parse rank/alpha/targets out of a loaded adapter file's `__metadata__`
/// (from
/// [`SafeTensors::metadata`](crate::format::safetensors::SafeTensors::metadata)).
/// The ONE parser: [`check_lora_metadata`] calls this then compares the
/// result against a caller's expected values, and
/// `VoxCpm2Model::load_lora_adapter` calls this to get the values it
/// applies with — neither hand-rolls its own parse.
///
/// Errors when `metadata` is EMPTY: an adapter saved before this crate
/// wrote LoRA config metadata carries no proof of its rank/alpha/targets.
/// Re-save the adapter through [`build_lora_metadata`] to attach metadata,
/// or verify rank/alpha/targets by hand before loading it. Also errors when
/// `lora_rank`/`lora_alpha`/`lora_targets` is missing or malformed, naming
/// every such field at once.
pub fn parse_lora_metadata(metadata: &HashMap<String, String>) -> Result<LoraMetadata> {
    if metadata.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "metadata",
            reason: "adapter file has no __metadata__ (it predates LoRA config metadata \
                     support); re-save it through build_lora_metadata to attach \
                     rank/alpha/targets, or verify them against this model by hand before \
                     loading"
                .into(),
        });
    }

    let mut problems = Vec::new();

    let rank = match metadata.get(LORA_METADATA_RANK_KEY) {
        Some(found) => match found.parse::<usize>() {
            Ok(v) => Some(v),
            Err(_) => {
                problems.push(format!(
                    "{LORA_METADATA_RANK_KEY}: not a valid integer: '{found}'"
                ));
                None
            }
        },
        None => {
            problems.push(format!("{LORA_METADATA_RANK_KEY}: missing from metadata"));
            None
        }
    };

    let alpha = match metadata.get(LORA_METADATA_ALPHA_KEY) {
        Some(found) => match found.parse::<f32>() {
            Ok(v) => Some(v),
            Err(_) => {
                problems.push(format!(
                    "{LORA_METADATA_ALPHA_KEY}: not a valid number: '{found}'"
                ));
                None
            }
        },
        None => {
            problems.push(format!("{LORA_METADATA_ALPHA_KEY}: missing from metadata"));
            None
        }
    };

    let targets = match metadata.get(LORA_METADATA_TARGETS_KEY) {
        Some(found) => {
            let names: Vec<String> = found
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            if names.is_empty() {
                problems.push(format!("{LORA_METADATA_TARGETS_KEY}: empty target list"));
                None
            } else {
                Some(names)
            }
        }
        None => {
            problems.push(format!(
                "{LORA_METADATA_TARGETS_KEY}: missing from metadata"
            ));
            None
        }
    };

    match (rank, alpha, targets) {
        (Some(rank), Some(alpha), Some(targets)) => Ok(LoraMetadata {
            rank,
            alpha,
            targets,
        }),
        _ => Err(Error::InvalidArgument {
            arg: "metadata",
            reason: format!("LoRA adapter metadata invalid: {}", problems.join("; ")),
        }),
    }
}

/// Check a loaded adapter file's `__metadata__` against the rank/alpha/
/// targets the caller is about to load it into. Parses via
/// [`parse_lora_metadata`], then compares.
///
/// Each disagreeing field is named individually, with both the expected
/// and the found value — never one generic "config mismatch" message.
/// `targets` is compared as a set: `--targets v_proj,q_proj` and
/// `--targets q_proj,v_proj` name the same adapted projections.
pub fn check_lora_metadata(
    metadata: &HashMap<String, String>,
    expected_rank: usize,
    expected_alpha: f32,
    expected_targets: &[String],
) -> Result<()> {
    let found = parse_lora_metadata(metadata)?;

    let mut mismatches = Vec::new();

    if found.rank != expected_rank {
        mismatches.push(format!(
            "{LORA_METADATA_RANK_KEY}: expected {expected_rank}, found {}",
            found.rank
        ));
    }

    if found.alpha != expected_alpha {
        mismatches.push(format!(
            "{LORA_METADATA_ALPHA_KEY}: expected {expected_alpha}, found {}",
            found.alpha
        ));
    }

    let mut found_sorted: Vec<&str> = found.targets.iter().map(String::as_str).collect();
    found_sorted.sort_unstable();
    let mut expected_sorted: Vec<&str> = expected_targets.iter().map(String::as_str).collect();
    expected_sorted.sort_unstable();
    if found_sorted != expected_sorted {
        mismatches.push(format!(
            "{LORA_METADATA_TARGETS_KEY}: expected {expected_sorted:?}, found {found_sorted:?}"
        ));
    }

    if mismatches.is_empty() {
        return Ok(());
    }

    Err(Error::InvalidArgument {
        arg: "metadata",
        reason: format!("LoRA adapter metadata mismatch: {}", mismatches.join("; ")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_then_check_round_trips() {
        let targets = vec!["q_proj".to_string(), "v_proj".to_string()];
        let meta = build_lora_metadata(16, 32.0, &targets);
        assert!(check_lora_metadata(&meta, 16, 32.0, &targets).is_ok());
    }

    #[test]
    fn check_accepts_reordered_targets() {
        let saved = vec!["q_proj".to_string(), "v_proj".to_string()];
        let meta = build_lora_metadata(16, 32.0, &saved);
        let expected = vec!["v_proj".to_string(), "q_proj".to_string()];
        assert!(check_lora_metadata(&meta, 16, 32.0, &expected).is_ok());
    }

    #[test]
    fn check_rejects_empty_metadata() {
        let meta = HashMap::new();
        let err = check_lora_metadata(&meta, 16, 32.0, &["q_proj".to_string()]).unwrap_err();
        assert!(err.to_string().contains("__metadata__"), "got {err}");
    }

    #[test]
    fn check_reports_rank_mismatch() {
        let targets = vec!["q_proj".to_string()];
        let meta = build_lora_metadata(16, 32.0, &targets);
        let err = check_lora_metadata(&meta, 8, 32.0, &targets).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("lora_rank"), "got {message}");
        assert!(message.contains("expected 8"), "got {message}");
        assert!(message.contains("found 16"), "got {message}");
    }

    #[test]
    fn check_reports_alpha_mismatch() {
        let targets = vec!["q_proj".to_string()];
        let meta = build_lora_metadata(16, 32.0, &targets);
        let err = check_lora_metadata(&meta, 16, 64.0, &targets).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("lora_alpha"), "got {message}");
        assert!(message.contains("expected 64"), "got {message}");
        assert!(message.contains("found 32"), "got {message}");
    }

    #[test]
    fn check_reports_targets_mismatch() {
        let saved = vec!["q_proj".to_string()];
        let meta = build_lora_metadata(16, 32.0, &saved);
        let expected = vec!["v_proj".to_string()];
        let err = check_lora_metadata(&meta, 16, 32.0, &expected).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("lora_targets"), "got {message}");
        assert!(message.contains("v_proj"), "got {message}");
        assert!(message.contains("q_proj"), "got {message}");
    }
}
