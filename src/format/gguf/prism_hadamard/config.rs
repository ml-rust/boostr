//! `PrismHadamardConfig`: a typed, validated parse of the PrismML
//! activation-rotation contract.
//!
//! Weights in `weight_names` were quantized in a rotated basis: before the
//! matmul, the activation is sign-flipped then transformed per `block_size`
//! segment. `inverse_weight_names` store rotated rows: after lookup,
//! transform then sign-flip.
//!
//! Ports the checks in the PrismML llama.cpp fork's
//! `llama_model_base::load_hparams` (`llama-model.cpp` lines ~1194-1330).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::block::parse_block_size;
use super::signs::parse_signs;
use super::weights::{
    parse_inverse_weight_names, parse_weight_names_raw, validate_foldable_and_dedupe,
};
use crate::error::{Error, Result};
use crate::format::gguf::metadata::GgufMetadata;

pub(super) const KEY_VERSION: &str = "prism.hadamard.version";
pub(super) const KEY_BLOCK_SIZE: &str = "prism.hadamard.block_size";
pub(super) const KEY_TRANSFORM: &str = "prism.hadamard.transform";
pub(super) const KEY_AXIS: &str = "prism.hadamard.axis";
pub(super) const KEY_SIGN_MODE: &str = "prism.hadamard.sign_mode";
pub(super) const KEY_WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
pub(super) const KEY_SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
pub(super) const KEY_SIGN_VALUES: &str = "prism.hadamard.sign_values";
pub(super) const KEY_INVERSE_WEIGHT_NAMES: &str = "prism.hadamard.inverse_weight_names";
pub(super) const KEY_GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";

pub(super) const TRANSFORM_NAME: &str = "normalized-sylvester-walsh-hadamard";
pub(super) const AXIS_NAME: &str = "input-last-dimension";

pub(super) fn model_error(reason: String) -> Error {
    Error::ModelError { reason }
}

pub(super) fn missing_key(key: &str) -> Error {
    model_error(format!("{key} is missing"))
}

/// Sign-vector source for the activation-rotation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignMode {
    /// No sign flip; the block transform alone rotates the activation.
    Identity,
    /// Sign vector looked up per input width from the parsed sign table.
    Explicit,
}

/// Activation rotation contract stored by the PrismML llama.cpp fork.
///
/// Weights in `weight_names` were quantized in a rotated basis. Before the
/// matmul, the activation is sign-flipped then transformed per `block_size`
/// segment. `inverse_weight_names` store rotated rows: after lookup,
/// transform then sign-flip.
///
/// Serializes as part of `UniversalConfig` so a config written to disk
/// carries the rotation contract with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismHadamardConfig {
    pub block_size: usize,
    pub sign_mode: SignMode,
    /// Sign vector per input width. Empty when `sign_mode` is `Identity`.
    signs_by_width: BTreeMap<usize, Vec<i8>>,
    weight_names: BTreeSet<String>,
    inverse_weight_names: BTreeSet<String>,
    pub gdn_v_grouped: bool,
}

impl PrismHadamardConfig {
    /// `Ok(None)` when `prism.hadamard.version` is absent.
    pub fn from_metadata(meta: &GgufMetadata) -> Result<Option<Self>> {
        let Some(version) = meta.get_u32(KEY_VERSION) else {
            return Ok(None);
        };
        if version != 1 {
            return Err(model_error(format!("unsupported {KEY_VERSION}: {version}")));
        }

        let block_size = parse_block_size(meta)?;

        let transform = meta
            .get_string(KEY_TRANSFORM)
            .ok_or_else(|| missing_key(KEY_TRANSFORM))?;
        if transform != TRANSFORM_NAME {
            return Err(model_error(format!(
                "unsupported {KEY_TRANSFORM}: '{transform}'"
            )));
        }

        let axis = meta
            .get_string(KEY_AXIS)
            .ok_or_else(|| missing_key(KEY_AXIS))?;
        if axis != AXIS_NAME {
            return Err(model_error(format!("unsupported {KEY_AXIS}: '{axis}'")));
        }

        let sign_mode_str = meta
            .get_string(KEY_SIGN_MODE)
            .ok_or_else(|| missing_key(KEY_SIGN_MODE))?;
        let sign_mode = match sign_mode_str {
            "identity" => SignMode::Identity,
            "explicit" => SignMode::Explicit,
            other => {
                return Err(model_error(format!(
                    "unsupported {KEY_SIGN_MODE}: '{other}'"
                )));
            }
        };

        let signs_by_width = match sign_mode {
            SignMode::Identity => BTreeMap::new(),
            SignMode::Explicit => parse_signs(meta, block_size)?,
        };

        // Cross-check weight/inverse overlap on the raw name lists, before
        // the foldable-path check narrows `weight_names` — an entry named in
        // both places is a broken file regardless of whether it is also on
        // the verified matmul path.
        let weight_names_raw = parse_weight_names_raw(meta)?;
        let inverse_names_raw = parse_inverse_weight_names(meta)?;
        for name in &inverse_names_raw {
            if weight_names_raw.contains(name) {
                return Err(model_error(format!(
                    "prism.hadamard: '{name}' is in both {KEY_WEIGHT_NAMES} and \
                     {KEY_INVERSE_WEIGHT_NAMES}"
                )));
            }
        }
        let weight_names = validate_foldable_and_dedupe(weight_names_raw)?;
        let inverse_weight_names: BTreeSet<String> = inverse_names_raw.into_iter().collect();

        let gdn_v_grouped = meta.get_bool(KEY_GDN_V_GROUPED).unwrap_or(false);

        Ok(Some(Self {
            block_size,
            sign_mode,
            signs_by_width,
            weight_names,
            inverse_weight_names,
            gdn_v_grouped,
        }))
    }

    /// True when `tensor_name` was quantized in the rotated basis.
    pub fn rotates(&self, tensor_name: &str) -> bool {
        self.weight_names.contains(tensor_name)
    }

    /// True when `tensor_name` stores rotated rows needing the inverse
    /// transform after lookup.
    pub fn inverts(&self, tensor_name: &str) -> bool {
        self.inverse_weight_names.contains(tensor_name)
    }

    /// Sign vector for an input width. `Identity` mode returns `Ok(None)`.
    ///
    /// Explicit mode errors when the width has no entry — the fork silently
    /// falls back to identity there; boostr treats it as a broken file.
    pub fn signs_for_width(&self, width: usize) -> Result<Option<&[i8]>> {
        match self.sign_mode {
            SignMode::Identity => Ok(None),
            SignMode::Explicit => match self.signs_by_width.get(&width) {
                Some(signs) => Ok(Some(signs.as_slice())),
                None => Err(model_error(format!(
                    "prism.hadamard: no sign vector for width {width}; declared widths: {:?}",
                    self.signs_by_width.keys().collect::<Vec<_>>()
                ))),
            },
        }
    }

    /// Names of weights quantized in the rotated basis.
    pub fn weight_names(&self) -> impl Iterator<Item = &str> {
        self.weight_names.iter().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::GgufValue;

    fn meta(pairs: Vec<(&str, GgufValue)>) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        for (k, v) in pairs {
            m.kv.insert(k.to_string(), v);
        }
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

    fn ints(vals: &[i32]) -> GgufValue {
        GgufValue::Array(vals.iter().map(|v| GgufValue::Int32(*v)).collect())
    }

    /// Two widths (4 and 8), both multiples of `block_size` 4, covering
    /// `sign_values` exactly (4 + 8 = 12 values).
    fn valid_pairs() -> Vec<(&'static str, GgufValue)> {
        vec![
            (KEY_VERSION, GgufValue::Uint32(1)),
            (KEY_BLOCK_SIZE, GgufValue::Uint32(4)),
            (KEY_TRANSFORM, GgufValue::String(TRANSFORM_NAME.to_string())),
            (KEY_AXIS, GgufValue::String(AXIS_NAME.to_string())),
            (KEY_SIGN_MODE, GgufValue::String("explicit".to_string())),
            (
                KEY_WEIGHT_NAMES,
                strings(&["output.weight", "blk.0.attn_q.weight"]),
            ),
            (KEY_SIGN_WIDTHS, ints(&[4, 8])),
            (
                KEY_SIGN_VALUES,
                ints(&[1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1]),
            ),
        ]
    }

    fn replace(pairs: &mut Vec<(&'static str, GgufValue)>, key: &'static str, value: GgufValue) {
        pairs.retain(|(k, _)| *k != key);
        pairs.push((key, value));
    }

    #[test]
    fn absent_version_returns_none() {
        let m = meta(vec![]);
        assert!(PrismHadamardConfig::from_metadata(&m).unwrap().is_none());
    }

    #[test]
    fn minimal_valid_explicit_config() {
        let m = meta(valid_pairs());
        let cfg = PrismHadamardConfig::from_metadata(&m).unwrap().unwrap();
        assert_eq!(cfg.block_size, 4);
        assert_eq!(cfg.sign_mode, SignMode::Explicit);
        assert!(cfg.rotates("output.weight"));
        assert!(cfg.rotates("blk.0.attn_q.weight"));
        assert!(!cfg.rotates("blk.0.attn_k.weight"));
        assert!(!cfg.gdn_v_grouped);
        assert_eq!(cfg.weight_names().count(), 2);
        assert_eq!(cfg.signs_for_width(4).unwrap().unwrap().len(), 4);
        assert_eq!(cfg.signs_for_width(8).unwrap().unwrap().len(), 8);
    }

    #[test]
    fn identity_mode_returns_no_signs() {
        let mut pairs = valid_pairs();
        pairs
            .retain(|(k, _)| *k != KEY_SIGN_MODE && *k != KEY_SIGN_WIDTHS && *k != KEY_SIGN_VALUES);
        pairs.push((KEY_SIGN_MODE, GgufValue::String("identity".to_string())));
        let cfg = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap()
            .unwrap();
        assert_eq!(cfg.sign_mode, SignMode::Identity);
        assert_eq!(cfg.signs_for_width(4).unwrap(), None);
    }

    #[test]
    fn signs_for_width_unknown_width_errors() {
        let cfg = PrismHadamardConfig::from_metadata(&meta(valid_pairs()))
            .unwrap()
            .unwrap();
        let err = cfg.signs_for_width(99).unwrap_err().to_string();
        assert!(err.contains("99"));
    }

    #[test]
    fn version_mismatch_errors() {
        let mut pairs = valid_pairs();
        replace(&mut pairs, KEY_VERSION, GgufValue::Uint32(2));
        let err = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KEY_VERSION));
    }

    #[test]
    fn transform_mismatch_errors() {
        let mut pairs = valid_pairs();
        replace(
            &mut pairs,
            KEY_TRANSFORM,
            GgufValue::String("something-else".to_string()),
        );
        let err = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KEY_TRANSFORM));
    }

    #[test]
    fn axis_mismatch_errors() {
        let mut pairs = valid_pairs();
        replace(
            &mut pairs,
            KEY_AXIS,
            GgufValue::String("wrong-axis".to_string()),
        );
        let err = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KEY_AXIS));
    }

    #[test]
    fn sign_mode_invalid_errors() {
        let mut pairs = valid_pairs();
        replace(
            &mut pairs,
            KEY_SIGN_MODE,
            GgufValue::String("bogus".to_string()),
        );
        let err = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KEY_SIGN_MODE));
    }

    #[test]
    fn name_in_both_weight_and_inverse_errors() {
        let mut pairs = valid_pairs();
        replace(
            &mut pairs,
            KEY_WEIGHT_NAMES,
            strings(&["token_embd.weight"]),
        );
        pairs.push((KEY_INVERSE_WEIGHT_NAMES, strings(&["token_embd.weight"])));
        let err = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap_err()
            .to_string();
        assert!(err.contains("token_embd.weight"));
    }

    #[test]
    fn serde_round_trip_keeps_names_and_signs() {
        let cfg = PrismHadamardConfig::from_metadata(&meta(valid_pairs()))
            .unwrap()
            .unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: PrismHadamardConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.block_size, 4);
        assert_eq!(back.sign_mode, SignMode::Explicit);
        assert!(back.rotates("output.weight"));
        assert!(back.rotates("blk.0.attn_q.weight"));
        assert_eq!(
            back.signs_for_width(8).unwrap(),
            cfg.signs_for_width(8).unwrap()
        );
    }

    #[test]
    fn gdn_v_grouped_true_is_read() {
        let mut pairs = valid_pairs();
        pairs.push((KEY_GDN_V_GROUPED, GgufValue::Bool(true)));
        let cfg = PrismHadamardConfig::from_metadata(&meta(pairs))
            .unwrap()
            .unwrap();
        assert!(cfg.gdn_v_grouped);
    }
}
