//! Shape and dtype checks on loaded optimizer state tensors.

use super::keys::fail;
use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Check that `t` holds a dtype the optimizer keeps state in.
///
/// State is a float dtype at least as wide as F32.
pub(crate) fn check_state_dtype<R: Runtime<DType = DType>>(key: &str, t: &Tensor<R>) -> Result<()> {
    let dtype = t.dtype();
    if dtype.is_float() && !dtype.is_narrow_float() {
        Ok(())
    } else {
        Err(fail(format!(
            "optimizer state key '{key}': dtype {dtype:?} is not a state dtype, expected F32 or F64"
        )))
    }
}

/// Check that `t` has the shape and dtype of `reference`.
pub(crate) fn check_matches<R: Runtime<DType = DType>>(
    key: &str,
    t: &Tensor<R>,
    reference_key: &str,
    reference: &Tensor<R>,
) -> Result<()> {
    if t.shape() != reference.shape() {
        return Err(fail(format!(
            "optimizer state key '{key}': shape {:?} differs from '{reference_key}' shape {:?}",
            t.shape(),
            reference.shape()
        )));
    }
    if t.dtype() != reference.dtype() {
        return Err(fail(format!(
            "optimizer state key '{key}': dtype {:?} differs from '{reference_key}' dtype {:?}",
            t.dtype(),
            reference.dtype()
        )));
    }
    Ok(())
}
