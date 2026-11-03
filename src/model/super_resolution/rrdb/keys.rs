//! Legacy ESRGAN key names: the `model.*` layout of old ESRGAN checkpoints
//! such as `2x_PureVision`, mapped to [`super::RrdbNet`]'s names.
//!
//! The map:
//!
//! | Legacy | RRDBNet |
//! | --- | --- |
//! | `model.0` | `conv_first` |
//! | `model.1.sub.{b}.RDB{d}.conv{c}.0` | `body.{b}.rdb{d}.conv{c}` |
//! | `model.1.sub.{num_block}` | `conv_body` |
//! | `model.3` | `conv_up1` |
//! | `model.6` | `conv_up2` |
//! | `model.8` | `conv_hr` |
//! | `model.10` | `conv_last` |
//!
//! Each name takes a `.weight` or `.bias` suffix. `b` is a decimal block
//! index, `d` is 1 to 3, `c` is 1 to 5. A block index with leading zeros
//! names the same block as without them, so `sub.00` and `sub.0` collide.

use crate::error::{Error, Result};
use crate::model::super_resolution::rrdb::config::{RDB_CONVS, RRDB_DENSE_BLOCKS};
use std::collections::{BTreeMap, HashMap};

/// The key whose presence marks a legacy ESRGAN state dict.
pub const LEGACY_ESRGAN_MARKER: &str = "model.0.weight";

/// Legacy prefixes of the convs outside the body, except `conv_body`.
const FIXED: [(&str, &str); 5] = [
    ("model.0", "conv_first"),
    ("model.3", "conv_up1"),
    ("model.6", "conv_up2"),
    ("model.8", "conv_hr"),
    ("model.10", "conv_last"),
];

/// Whether `names` hold [`LEGACY_ESRGAN_MARKER`].
pub fn is_legacy_esrgan<'a>(mut names: impl Iterator<Item = &'a str>) -> bool {
    names.any(|name| name == LEGACY_ESRGAN_MARKER)
}

/// The RRDBNet name of legacy key `key` in a network of `num_block` RRDBs,
/// or `None` when `key` is not a legacy RRDBNet key.
pub fn legacy_esrgan_target(key: &str, num_block: usize) -> Option<String> {
    if let Some(target) = dense_target(key) {
        return Some(target);
    }
    let (prefix, param) = key.rsplit_once('.')?;
    if param != "weight" && param != "bias" {
        return None;
    }
    let conv = if prefix == format!("model.1.sub.{num_block}") {
        "conv_body"
    } else {
        FIXED
            .iter()
            .find(|(legacy, _)| *legacy == prefix)
            .map(|(_, conv)| *conv)?
    };
    Some(format!("{conv}.{param}"))
}

/// Map every legacy key in `names` to its RRDBNet name.
///
/// The result maps source key to target name.
///
/// # Errors
///
/// - A key that is not a legacy RRDBNet key. The error names the key.
/// - Two keys that map to one target. The error names both keys and the
///   target.
pub fn legacy_esrgan_renames<'a>(
    names: impl Iterator<Item = &'a str>,
    num_block: usize,
) -> Result<HashMap<String, String>> {
    let mut by_target: BTreeMap<String, &str> = BTreeMap::new();
    for key in names {
        let target = legacy_esrgan_target(key, num_block).ok_or_else(|| Error::ModelError {
            reason: format!(
                "legacy ESRGAN key '{key}' is not a parameter of a {num_block}-block RRDBNet x2; \
                 use a 2x ESRGAN checkpoint with the RRDBNet layout"
            ),
        })?;
        if let Some(first) = by_target.insert(target.clone(), key) {
            return Err(Error::ModelError {
                reason: format!(
                    "legacy ESRGAN keys '{first}' and '{key}' both map to '{target}'; \
                     use the original model download"
                ),
            });
        }
    }
    Ok(by_target
        .into_iter()
        .map(|(target, key)| (key.to_string(), target))
        .collect())
}

/// `model.1.sub.{b}.RDB{d}.conv{c}.0.{weight|bias}` as
/// `body.{b}.rdb{d}.conv{c}.{weight|bias}`.
fn dense_target(key: &str) -> Option<String> {
    let rest = key.strip_prefix("model.1.sub.")?;
    let mut parts = rest.split('.');
    let (block, dense, conv, zero, param) = (
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
    );
    if parts.next().is_some() || zero != "0" || (param != "weight" && param != "bias") {
        return None;
    }
    let block = decimal(block)?;
    let dense = indexed(dense, "RDB", RRDB_DENSE_BLOCKS)?;
    let conv = indexed(conv, "conv", RDB_CONVS)?;
    Some(format!("body.{block}.rdb{dense}.conv{conv}.{param}"))
}

/// `text` as a decimal `usize`. ASCII digits only, no sign.
fn decimal(text: &str) -> Option<usize> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// The single digit after `stem` in `text`, when it lies in `1..=max`.
fn indexed(text: &str, stem: &str, max: usize) -> Option<usize> {
    let digit = text.strip_prefix(stem)?;
    if digit.len() != 1 {
        return None;
    }
    decimal(digit).filter(|d| (1..=max).contains(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_names_map() {
        let cases = [
            ("model.0.weight", "conv_first.weight"),
            ("model.1.sub.23.bias", "conv_body.bias"),
            ("model.3.weight", "conv_up1.weight"),
            ("model.6.bias", "conv_up2.bias"),
            ("model.8.weight", "conv_hr.weight"),
            ("model.10.bias", "conv_last.bias"),
        ];
        for (legacy, modern) in cases {
            assert_eq!(
                legacy_esrgan_target(legacy, 23).as_deref(),
                Some(modern),
                "{legacy}"
            );
        }
    }

    #[test]
    fn dense_names_map() {
        assert_eq!(
            legacy_esrgan_target("model.1.sub.22.RDB3.conv5.0.weight", 23).as_deref(),
            Some("body.22.rdb3.conv5.weight")
        );
        assert_eq!(
            legacy_esrgan_target("model.1.sub.0.RDB1.conv1.0.bias", 23).as_deref(),
            Some("body.0.rdb1.conv1.bias")
        );
    }

    #[test]
    fn conv_body_follows_num_block() {
        assert_eq!(
            legacy_esrgan_target("model.1.sub.1.weight", 1).as_deref(),
            Some("conv_body.weight")
        );
        assert_eq!(legacy_esrgan_target("model.1.sub.23.weight", 1), None);
    }

    #[test]
    fn other_keys_are_unknown() {
        for key in [
            "model.2.weight",
            "model.0.running_mean",
            "model.1.sub.0.RDB4.conv1.0.weight",
            "model.1.sub.0.RDB1.conv6.0.weight",
            "model.1.sub.0.RDB1.conv1.1.weight",
            "model.1.sub.0.RDB1.conv1.0.weight.extra",
            "model.1.sub.-1.RDB1.conv1.0.weight",
            "model.1.sub.x.RDB1.conv1.0.weight",
            "conv_first.weight",
        ] {
            assert_eq!(legacy_esrgan_target(key, 23), None, "{key}");
        }
    }

    #[test]
    fn unknown_key_is_named() {
        let err = legacy_esrgan_renames(["model.0.weight", "model.2.weight"].into_iter(), 23)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("'model.2.weight'"), "{err}");
    }

    #[test]
    fn duplicate_target_names_both_keys() {
        let err = legacy_esrgan_renames(
            [
                "model.1.sub.0.RDB1.conv1.0.weight",
                "model.1.sub.00.RDB1.conv1.0.weight",
            ]
            .into_iter(),
            23,
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(err.contains("'body.0.rdb1.conv1.weight'"), "{err}");
        assert!(
            err.contains("'model.1.sub.00.RDB1.conv1.0.weight'"),
            "{err}"
        );
    }

    #[test]
    fn marker_detects_legacy() {
        assert!(is_legacy_esrgan(
            ["model.1.sub.0.RDB1.conv1.0.weight", "model.0.weight"].into_iter()
        ));
        assert!(!is_legacy_esrgan(["conv_first.weight"].into_iter()));
    }
}
