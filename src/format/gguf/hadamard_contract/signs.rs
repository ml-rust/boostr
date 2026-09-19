//! Explicit-mode sign-vector parsing: `sign_widths` + `sign_values` into a
//! per-width lookup table.

use std::collections::BTreeMap;

use super::config::{KEY_BLOCK_SIZE, KEY_SIGN_VALUES, KEY_SIGN_WIDTHS, missing_key, model_error};
use crate::error::Result;
use crate::format::gguf::metadata::GgufMetadata;

/// Consumes `sign_values` in `sign_widths` order. Mirrors the fork: each
/// width must be positive, a multiple of `block_size`, and in bounds; each
/// value must be `+1`/`-1`; the whole `sign_values` array is consumed
/// exactly; no width repeats.
pub(super) fn parse_signs(
    meta: &GgufMetadata,
    block_size: usize,
) -> Result<BTreeMap<usize, Vec<i8>>> {
    let widths = meta
        .get_i64_array(KEY_SIGN_WIDTHS)
        .ok_or_else(|| missing_key(KEY_SIGN_WIDTHS))?;
    let values = meta
        .get_i64_array(KEY_SIGN_VALUES)
        .ok_or_else(|| missing_key(KEY_SIGN_VALUES))?;

    let mut table = BTreeMap::new();
    let mut off = 0usize;
    for width in widths {
        if width <= 0 {
            return Err(model_error(format!(
                "invalid {KEY_SIGN_WIDTHS} entry: {width} (must be positive)"
            )));
        }
        let width = width as usize;
        if !width.is_multiple_of(block_size) {
            return Err(model_error(format!(
                "invalid {KEY_SIGN_WIDTHS} entry: {width} is not a multiple of \
                 {KEY_BLOCK_SIZE} ({block_size})"
            )));
        }
        if off + width > values.len() {
            return Err(model_error(format!(
                "{KEY_SIGN_VALUES} is too short for {KEY_SIGN_WIDTHS} entry {width}: \
                 need {width} elements starting at offset {off}, {KEY_SIGN_VALUES} has \
                 {} elements",
                values.len()
            )));
        }

        let mut signs = Vec::with_capacity(width);
        for &v in &values[off..off + width] {
            if v != 1 && v != -1 {
                return Err(model_error(format!(
                    "invalid {KEY_SIGN_VALUES} entry: {v} (must be +1 or -1)"
                )));
            }
            signs.push(v as i8);
        }
        if table.insert(width, signs).is_some() {
            return Err(model_error(format!(
                "duplicate {KEY_SIGN_WIDTHS} entry: {width}"
            )));
        }
        off += width;
    }

    if off != values.len() {
        return Err(model_error(format!(
            "{KEY_SIGN_VALUES} length mismatch: {KEY_SIGN_WIDTHS} consumes {off} elements, \
             {KEY_SIGN_VALUES} has {} elements",
            values.len()
        )));
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::GgufValue;

    fn meta(widths: &[i32], values: &[i32]) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        m.kv.insert(
            KEY_SIGN_WIDTHS.to_string(),
            GgufValue::Array(widths.iter().map(|v| GgufValue::Int32(*v)).collect()),
        );
        m.kv.insert(
            KEY_SIGN_VALUES.to_string(),
            GgufValue::Array(values.iter().map(|v| GgufValue::Int32(*v)).collect()),
        );
        m
    }

    #[test]
    fn sign_widths_missing_errors() {
        let m = GgufMetadata::default();
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_WIDTHS));
    }

    #[test]
    fn sign_values_missing_errors() {
        let mut m = GgufMetadata::default();
        m.kv.insert(
            KEY_SIGN_WIDTHS.to_string(),
            GgufValue::Array(vec![GgufValue::Int32(4)]),
        );
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_VALUES));
    }

    #[test]
    fn width_not_positive_errors() {
        let m = meta(&[0], &[]);
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_WIDTHS));
    }

    #[test]
    fn width_not_multiple_of_block_size_errors() {
        let m = meta(&[6], &[1, -1, 1, -1, 1, -1]);
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_BLOCK_SIZE));
    }

    #[test]
    fn width_out_of_bounds_errors() {
        let m = meta(&[8], &[1, -1, 1, -1]);
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_VALUES));
    }

    #[test]
    fn value_not_pm1_errors() {
        let m = meta(&[4], &[1, -1, 2, -1]);
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_VALUES));
    }

    #[test]
    fn length_mismatch_errors() {
        let m = meta(&[4], &[1, -1, 1, -1, 1]);
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_VALUES));
    }

    #[test]
    fn duplicate_width_errors() {
        let m = meta(&[4, 4], &[1, -1, 1, -1, 1, -1, 1, -1]);
        let err = parse_signs(&m, 4).unwrap_err().to_string();
        assert!(err.contains(KEY_SIGN_WIDTHS));
    }

    #[test]
    fn valid_two_widths_parsed() {
        let m = meta(&[4, 8], &[1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1]);
        let table = parse_signs(&m, 4).unwrap();
        assert_eq!(table.get(&4).unwrap().len(), 4);
        assert_eq!(table.get(&8).unwrap().len(), 8);
    }
}
