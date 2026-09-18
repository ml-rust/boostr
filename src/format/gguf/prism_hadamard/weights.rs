//! Weight-name validation for the PrismML Hadamard contract: which tensors
//! are foldable into a Hadamard-aware matmul, and the single verified
//! inverse-after-lookup table.

use std::collections::BTreeSet;

use super::config::{KEY_INVERSE_WEIGHT_NAMES, KEY_WEIGHT_NAMES, missing_key, model_error};
use crate::error::Result;
use crate::format::gguf::metadata::GgufMetadata;

const INVERSE_TABLE_NAME: &str = "token_embd.weight";

/// `<suffix>` values valid in `blk.<n>.<suffix>.weight`, plus the bare
/// `output.weight` name — the fork's verified Hadamard-aware matmul path.
const FOLDABLE_SUFFIXES: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "ssm_out",
];

/// Reads `weight_names`; missing or empty is an error. Foldability is not
/// checked here so the caller can cross-check against `inverse_weight_names`
/// before rejecting an unfoldable name.
pub(super) fn parse_weight_names_raw(meta: &GgufMetadata) -> Result<Vec<String>> {
    let raw = meta
        .get_string_array(KEY_WEIGHT_NAMES)
        .ok_or_else(|| missing_key(KEY_WEIGHT_NAMES))?;
    if raw.is_empty() {
        return Err(model_error(format!("{KEY_WEIGHT_NAMES} is empty")));
    }
    Ok(raw)
}

/// Validates every name is on the fork's verified matmul path and no name
/// repeats, building the lookup set.
pub(super) fn validate_foldable_and_dedupe(raw: Vec<String>) -> Result<BTreeSet<String>> {
    let mut set = BTreeSet::new();
    for name in raw {
        if !is_foldable_weight(&name) {
            return Err(model_error(format!(
                "prism.hadamard: weight '{name}' in {KEY_WEIGHT_NAMES} is not on a verified \
                 Hadamard-aware matmul path"
            )));
        }
        if !set.insert(name.clone()) {
            return Err(model_error(format!(
                "duplicate {KEY_WEIGHT_NAMES} entry: '{name}'"
            )));
        }
    }
    Ok(set)
}

/// `Ok([])` when the key is absent or an empty array. When present and
/// non-empty, the fork applies the inverse transform only to the
/// token-embedding lookup, so the array must be exactly
/// `["token_embd.weight"]`.
pub(super) fn parse_inverse_weight_names(meta: &GgufMetadata) -> Result<Vec<String>> {
    let Some(raw) = meta.get_string_array(KEY_INVERSE_WEIGHT_NAMES) else {
        return Ok(Vec::new());
    };
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    if raw.len() != 1 || raw[0] != INVERSE_TABLE_NAME {
        return Err(model_error(format!(
            "{KEY_INVERSE_WEIGHT_NAMES} must be exactly [\"{INVERSE_TABLE_NAME}\"], got {raw:?}"
        )));
    }
    Ok(raw)
}

/// True for `output.weight` or `blk.<n>.<suffix>.weight` with `<suffix>` in
/// [`FOLDABLE_SUFFIXES`]. Mirrors the fork's `is_foldable_weight`.
fn is_foldable_weight(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let (n, suffix_with_dot) = rest.split_at(dot);
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let Some(suffix) = suffix_with_dot[1..].strip_suffix(".weight") else {
        return false;
    };
    FOLDABLE_SUFFIXES.contains(&suffix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::GgufValue;

    fn meta_with(key: &str, value: GgufValue) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        m.kv.insert(key.to_string(), value);
        m
    }

    fn strings(names: &[&str]) -> GgufValue {
        GgufValue::Array(
            names
                .iter()
                .map(|s| GgufValue::String((*s).to_string()))
                .collect(),
        )
    }

    #[test]
    fn weight_names_missing_errors() {
        let m = GgufMetadata::default();
        let err = parse_weight_names_raw(&m).unwrap_err().to_string();
        assert!(err.contains(KEY_WEIGHT_NAMES));
    }

    #[test]
    fn weight_names_empty_errors() {
        let m = meta_with(KEY_WEIGHT_NAMES, strings(&[]));
        let err = parse_weight_names_raw(&m).unwrap_err().to_string();
        assert!(err.contains(KEY_WEIGHT_NAMES));
    }

    #[test]
    fn unfoldable_weight_name_errors() {
        let raw = vec!["not.a.weight".to_string()];
        let err = validate_foldable_and_dedupe(raw).unwrap_err().to_string();
        assert!(err.contains("not.a.weight"));
    }

    #[test]
    fn duplicate_weight_name_errors() {
        let raw = vec!["output.weight".to_string(), "output.weight".to_string()];
        let err = validate_foldable_and_dedupe(raw).unwrap_err().to_string();
        assert!(err.contains(KEY_WEIGHT_NAMES));
    }

    #[test]
    fn all_foldable_kinds_accepted() {
        let mut raw = vec!["output.weight".to_string()];
        for kind in FOLDABLE_SUFFIXES {
            raw.push(format!("blk.7.{kind}.weight"));
        }
        assert!(validate_foldable_and_dedupe(raw).is_ok());
    }

    #[test]
    fn inverse_weight_names_absent_is_empty() {
        let m = GgufMetadata::default();
        assert!(parse_inverse_weight_names(&m).unwrap().is_empty());
    }

    #[test]
    fn inverse_weight_names_wrong_errors() {
        let m = meta_with(KEY_INVERSE_WEIGHT_NAMES, strings(&["blk.0.attn_q.weight"]));
        let err = parse_inverse_weight_names(&m).unwrap_err().to_string();
        assert!(err.contains(KEY_INVERSE_WEIGHT_NAMES));
    }

    #[test]
    fn inverse_weight_names_valid_is_accepted() {
        let m = meta_with(KEY_INVERSE_WEIGHT_NAMES, strings(&["token_embd.weight"]));
        assert_eq!(
            parse_inverse_weight_names(&m).unwrap(),
            vec!["token_embd.weight".to_string()]
        );
    }
}
