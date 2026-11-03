//! The `metadata.*` scalars some ESRGAN checkpoints carry, such as
//! `2x_PureVision`, checked against an [`RrdbConfig`].
//!
//! The fields follow the old ESRGAN architecture: `in_nc` counts the
//! channels after the pixel-unshuffle head, `scale` is the upscale of the
//! RRDB trunk, and `shuffle_factor` is the unshuffle that runs first. The
//! overall upscale is `scale / shuffle_factor`.

use crate::error::{Error, Result};
use crate::format::{PtScalar, TorchStateDict};
use crate::model::super_resolution::rrdb::config::{RRDB_UNSHUFFLE_FACTOR, RrdbConfig};

/// Fail unless every `metadata.*` field `state` holds agrees with `config`.
///
/// An absent field passes. A present field with another type or value is
/// an error that names the field, both values and the next action.
/// Checked fields:
///
/// - `in_nc` equals [`RrdbConfig::first_conv_in_channels`].
/// - `out_nc`, `num_filters` and `num_blocks` equal `num_out_ch`,
///   `num_feat` and `num_block`.
/// - `shuffle_factor` is 2.
/// - `scale / shuffle_factor` equals `config.scale`. A missing
///   `shuffle_factor` counts as 1.
/// - `plus` is `false`, `act` is `"leakyrelu"`, `upsampler` is `"upconv"`
///   and `mode` is `"CNA"`.
pub fn check_esrgan_metadata(state: &TorchStateDict, config: &RrdbConfig) -> Result<()> {
    let ints = [
        ("in_nc", config.first_conv_in_channels(), "num_in_ch · 4"),
        ("out_nc", config.num_out_ch, "num_out_ch"),
        ("num_filters", config.num_feat, "num_feat"),
        ("num_blocks", config.num_block, "num_block"),
        (
            "shuffle_factor",
            RRDB_UNSHUFFLE_FACTOR,
            "the x2 unshuffle head",
        ),
    ];
    for (field, want, source) in ints {
        if let Some(got) = int_field(state, field)?
            && got != want
        {
            return Err(mismatch(format!(
                "metadata.{field} is {got}, but {source} gives {want}"
            )));
        }
    }
    if let Some(scale) = int_field(state, "scale")? {
        let shuffle = int_field(state, "shuffle_factor")?.unwrap_or(1);
        if shuffle == 0 || !scale.is_multiple_of(shuffle) || scale / shuffle != config.scale {
            return Err(mismatch(format!(
                "metadata.scale {scale} with metadata.shuffle_factor {shuffle} is not an x{} network",
                config.scale
            )));
        }
    }
    if let Some(plus) = state.scalar("metadata.plus")
        && *plus != PtScalar::Bool(false)
    {
        return Err(mismatch(format!(
            "metadata.plus is {plus:?}; RRDBNet+ adds convs this network does not have"
        )));
    }
    let strings = [
        ("act", "leakyrelu"),
        ("upsampler", "upconv"),
        ("mode", "CNA"),
    ];
    for (field, want) in strings {
        let key = format!("metadata.{field}");
        match state.scalar(&key) {
            None => {}
            Some(PtScalar::Str(got)) if got == want => {}
            Some(got) => {
                return Err(mismatch(format!("{key} is {got:?}, expected \"{want}\"")));
            }
        }
    }
    Ok(())
}

/// `metadata.{field}` as a non-negative int, `None` when absent.
fn int_field(state: &TorchStateDict, field: &str) -> Result<Option<usize>> {
    let key = format!("metadata.{field}");
    match state.scalar(&key) {
        None => Ok(None),
        Some(PtScalar::Int(v)) => usize::try_from(*v)
            .map(Some)
            .map_err(|_| mismatch(format!("{key} is {v}; it must be a non-negative int"))),
        Some(other) => Err(mismatch(format!("{key} is {other:?}; it must be an int"))),
    }
}

fn mismatch(detail: String) -> Error {
    Error::ModelError {
        reason: format!(
            "checkpoint metadata does not match the RRDBNet config: {detail}; \
             pass the config the checkpoint was trained with, or use an x2 RRDBNet checkpoint"
        ),
    }
}
