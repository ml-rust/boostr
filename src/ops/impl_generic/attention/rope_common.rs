//! Shared validation and preparation helper for all RoPE variants.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_cast, var_narrow};
use numr::ops::{ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Output of `validate_and_prepare`: (shape, seq_len, half_d, cos_narrowed, sin_narrowed).
pub(super) type PrepareOutput<R> = (Vec<usize>, usize, usize, Var<R>, Var<R>);

/// Validate inputs common to all RoPE variants.
/// Returns `(shape, seq_len, half_d, cos_narrowed, sin_narrowed)` with caches
/// reshaped to `[1, 1, S, D/2]` for broadcasting.
///
/// If cos/sin caches have a different dtype from x (e.g. F32 caches with BF16 input),
/// they are cast to match x's dtype.
pub(super) fn validate_and_prepare<R, C>(
    client: &C,
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
) -> Result<PrepareOutput<R>>
where
    R: Runtime<DType = numr::dtype::DType>,
    C: RuntimeClient<R> + TypeConversionOps<R>,
    R::Client: RuntimeClient<R> + ShapeOps<R> + TensorOps<R> + TypeConversionOps<R>,
{
    let shape = x.tensor().shape().to_vec();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected 4D [B, H, S, D], got {}D", shape.len()),
        });
    }

    let d = shape[3];
    if d % 2 != 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("head dim D={} must be even for RoPE", d),
        });
    }

    let half_d = d / 2;
    let seq_len = shape[2];

    // Validate cos/sin cache shapes
    let cos_shape = cos_cache.tensor().shape();
    let sin_shape = sin_cache.tensor().shape();
    if cos_shape.len() != 2 || cos_shape[1] != half_d {
        return Err(Error::InvalidArgument {
            arg: "cos_cache",
            reason: format!("expected [S, {}], got {:?}", half_d, cos_shape),
        });
    }
    if sin_shape.len() != 2 || sin_shape[1] != half_d {
        return Err(Error::InvalidArgument {
            arg: "sin_cache",
            reason: format!("expected [S, {}], got {:?}", half_d, sin_shape),
        });
    }

    // Cast cos/sin to match x's dtype if needed
    let x_dtype = x.tensor().dtype();
    let cos_matched = if cos_cache.tensor().dtype() != x_dtype {
        var_cast(cos_cache, x_dtype, client).map_err(Error::Numr)?
    } else {
        cos_cache.clone()
    };
    let sin_matched = if sin_cache.tensor().dtype() != x_dtype {
        var_cast(sin_cache, x_dtype, client).map_err(Error::Numr)?
    } else {
        sin_cache.clone()
    };

    // Narrow cos/sin cache from [max_S, D/2] to [S, D/2] if needed
    let cos_narrowed = if cos_shape[0] > seq_len {
        var_narrow(&cos_matched, 0, 0, seq_len).map_err(Error::Numr)?
    } else {
        cos_matched
    };
    let sin_narrowed = if sin_shape[0] > seq_len {
        var_narrow(&sin_matched, 0, 0, seq_len).map_err(Error::Numr)?
    } else {
        sin_matched
    };

    // Reshape to [1, 1, S, D/2] — broadcasting handles B and H
    let cos_reshaped = numr::autograd::var_reshape(&cos_narrowed, &[1, 1, seq_len, half_d])
        .map_err(Error::Numr)?;
    let sin_reshaped = numr::autograd::var_reshape(&sin_narrowed, &[1, 1, seq_len, half_d])
        .map_err(Error::Numr)?;

    Ok((shape, seq_len, half_d, cos_reshaped, sin_reshaped))
}
