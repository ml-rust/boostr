//! Operand checks for the tracked causal conv.

use crate::error::{Error, Result};
use crate::nn::segment_masks::SegmentMasks;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Return `(batch, seq, channels, kernel)` after checking every operand.
pub(super) fn check_shapes<R: Runtime>(
    x: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
) -> Result<(usize, usize, usize, usize)> {
    let xs = x.shape();
    if xs.len() != 3 || xs[1] == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [batch, seq >= 1, channels], got {xs:?}"),
        });
    }
    let (batch, seq, channels) = (xs[0], xs[1], xs[2]);
    let ws = weight.shape();
    if ws.len() != 3 || ws[0] != channels || ws[1] != 1 || ws[2] == 0 {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!("expected [{channels}, 1, kernel >= 1], got {ws:?}"),
        });
    }
    if let Some(bias) = bias
        && bias.shape() != [channels]
    {
        return Err(Error::InvalidArgument {
            arg: "bias",
            reason: format!("expected [{channels}], got {:?}", bias.shape()),
        });
    }
    Ok((batch, seq, channels, ws[2]))
}

/// Check that `masks` covers a `[batch, seq]` input of `dtype` and `kernel` taps.
pub(super) fn check_segments<R: Runtime<DType = DType>>(
    masks: &SegmentMasks<R>,
    batch: usize,
    seq: usize,
    kernel: usize,
    dtype: DType,
) -> Result<()> {
    if masks.dims() != (batch, seq) {
        return Err(Error::InvalidArgument {
            arg: "segments",
            reason: format!("expected [{batch}, {seq}], got {:?}", masks.dims()),
        });
    }
    if masks.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg: "segments",
            reason: format!("expected dtype {dtype:?}, got {:?}", masks.dtype()),
        });
    }
    if masks.max_kernel() < kernel {
        return Err(Error::InvalidArgument {
            arg: "segments",
            reason: format!(
                "masks cover kernel {}, the conv needs {kernel}",
                masks.max_kernel()
            ),
        });
    }
    Ok(())
}
