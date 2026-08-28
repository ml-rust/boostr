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

/// Check a loaded adapter file's `__metadata__` (from
/// [`SafeTensors::metadata`](crate::format::safetensors::SafeTensors::metadata))
/// against the rank/alpha/targets the caller is about to load it into.
///
/// Errors when `metadata` is EMPTY: an adapter saved before this crate
/// wrote LoRA config metadata carries no proof it matches the model, and a
/// rank/alpha mismatch between training and inference is a top real-world
/// LoRA failure mode — so this refuses to pass an unproven file rather than
/// allow a silent bypass. Re-save the adapter through
/// [`build_lora_metadata`] to attach metadata, or verify rank/alpha/targets
/// by hand before loading it.
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

    let mut mismatches = Vec::new();

    match metadata.get(LORA_METADATA_RANK_KEY) {
        Some(found) => match found.parse::<usize>() {
            Ok(found_rank) if found_rank != expected_rank => mismatches.push(format!(
                "{LORA_METADATA_RANK_KEY}: expected {expected_rank}, found {found_rank}"
            )),
            Ok(_) => {}
            Err(_) => mismatches.push(format!(
                "{LORA_METADATA_RANK_KEY}: not a valid integer: '{found}'"
            )),
        },
        None => mismatches.push(format!("{LORA_METADATA_RANK_KEY}: missing from metadata")),
    }

    match metadata.get(LORA_METADATA_ALPHA_KEY) {
        Some(found) => match found.parse::<f32>() {
            Ok(found_alpha) if found_alpha != expected_alpha => mismatches.push(format!(
                "{LORA_METADATA_ALPHA_KEY}: expected {expected_alpha}, found {found_alpha}"
            )),
            Ok(_) => {}
            Err(_) => mismatches.push(format!(
                "{LORA_METADATA_ALPHA_KEY}: not a valid number: '{found}'"
            )),
        },
        None => mismatches.push(format!("{LORA_METADATA_ALPHA_KEY}: missing from metadata")),
    }

    match metadata.get(LORA_METADATA_TARGETS_KEY) {
        Some(found) => {
            let mut found_sorted: Vec<&str> = found.split(',').filter(|s| !s.is_empty()).collect();
            found_sorted.sort_unstable();
            let mut expected_sorted: Vec<&str> =
                expected_targets.iter().map(String::as_str).collect();
            expected_sorted.sort_unstable();
            if found_sorted != expected_sorted {
                mismatches.push(format!(
                    "{LORA_METADATA_TARGETS_KEY}: expected {expected_sorted:?}, found \
                     {found_sorted:?}"
                ));
            }
        }
        None => mismatches.push(format!(
            "{LORA_METADATA_TARGETS_KEY}: missing from metadata"
        )),
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
