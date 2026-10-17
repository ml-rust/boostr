//! The `__step` entry: encode and decode the global step counter.

use super::keys::{STEP_KEY, fail};
use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

/// Largest step that F64 stores exactly.
const MAX_EXACT_STEP: u64 = 1 << 53;

/// Write `timestep` under `__step` as a 1-element F64 tensor on `device`.
///
/// F64 is the carrier because WebGPU has no 64-bit integer dtype.
pub(crate) fn put_step<R: Runtime<DType = DType>>(
    out: &mut HashMap<String, Tensor<R>>,
    timestep: u64,
    device: &R::Device,
) -> Result<()> {
    if timestep > MAX_EXACT_STEP {
        return Err(fail(format!(
            "optimizer state key '{STEP_KEY}': step {timestep} exceeds {MAX_EXACT_STEP}, the largest exact F64 integer"
        )));
    }
    let tensor = Tensor::<R>::from_slice(&[timestep as f64], &[1], device)?;
    out.insert(STEP_KEY.to_string(), tensor);
    Ok(())
}

/// Read the global step from `state`.
///
/// `has_params` tells whether per-parameter keys are present. A missing
/// `__step` is an error then, and step 0 otherwise.
pub(crate) fn read_step<R: Runtime<DType = DType>>(
    state: &HashMap<String, Tensor<R>>,
    has_params: bool,
) -> Result<u64> {
    let Some(tensor) = state.get(STEP_KEY) else {
        return if has_params {
            Err(fail(format!(
                "optimizer state key '{STEP_KEY}' is missing while per-parameter keys are present"
            )))
        } else {
            Ok(0)
        };
    };
    if tensor.shape() != [1] {
        return Err(fail(format!(
            "optimizer state key '{STEP_KEY}': shape {:?} is not [1]",
            tensor.shape()
        )));
    }
    let tensor = tensor.contiguous()?;
    let value = match tensor.dtype() {
        DType::F64 => tensor.try_to_vec::<f64>()?.first().copied(),
        DType::I64 => tensor.try_to_vec::<i64>()?.first().map(|&v| v as f64),
        other => {
            return Err(fail(format!(
                "optimizer state key '{STEP_KEY}': dtype {other:?} is not F64 or I64"
            )));
        }
    };
    match value {
        Some(v) if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= MAX_EXACT_STEP as f64 => {
            Ok(v as u64)
        }
        other => Err(fail(format!(
            "optimizer state key '{STEP_KEY}': value {other:?} is not a step count"
        ))),
    }
}

/// Return an error when `state` holds `__step`.
///
/// AdaGrad and Sgd keep no global step, so the key marks a state from another optimizer.
pub(crate) fn reject_step<R: Runtime>(
    state: &HashMap<String, Tensor<R>>,
    optimizer: &str,
) -> Result<()> {
    if state.contains_key(STEP_KEY) {
        Err(fail(format!(
            "optimizer state key '{STEP_KEY}': {optimizer} keeps no global step"
        )))
    } else {
        Ok(())
    }
}
