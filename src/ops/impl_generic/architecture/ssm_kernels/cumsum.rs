//! SSD chunk cumsum kernel

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CumulativeOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute dA cumulative sum: softplus(dt + bias) * A → inclusive cumsum per chunk.
///
/// # Returns `(dt_out, dA_cumsum)`
/// - `dt_out`: `[batch, nheads, nchunks, chunk_size]`
/// - `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]`
#[allow(non_snake_case)]
pub fn ssd_chunk_cumsum_impl<R, C>(
    client: &C,
    dt: &Tensor<R>,
    a: &Tensor<R>,
    dt_bias: Option<&Tensor<R>>,
    chunk_size: usize,
    dt_softplus: bool,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ShapeOps<R>
        + ScalarOps<R>
        + ActivationOps<R>
        + ReduceOps<R>
        + CumulativeOps<R>
        + UtilityOps<R>
        + TensorOps<R>,
{
    // dt: [batch, seqlen, nheads]
    let dt_shape = dt.shape();
    if dt_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "dt",
            reason: format!(
                "expected 3D [batch, seqlen, nheads], got {}D",
                dt_shape.len()
            ),
        });
    }
    let batch = dt_shape[0];
    let seqlen = dt_shape[1];
    let nheads = dt_shape[2];
    let nchunks = seqlen.div_ceil(chunk_size);

    // Pad seqlen to nchunks * chunk_size if needed
    let padded_len = nchunks * chunk_size;
    let dt_padded = if padded_len > seqlen {
        let pad_size = padded_len - seqlen;
        let device = dt.device();
        let pad = Tensor::<R>::zeros(&[batch, pad_size, nheads], dt.dtype(), device);
        client.cat(&[dt, &pad], 1).map_err(Error::Numr)?
    } else {
        dt.clone()
    };

    // Add bias: dt + dt_bias [nheads] broadcast
    let dt_biased = if let Some(bias) = dt_bias {
        let bias_broad = bias.reshape(&[1, 1, nheads]).map_err(Error::Numr)?;
        client.add(&dt_padded, &bias_broad).map_err(Error::Numr)?
    } else {
        dt_padded
    };

    // Apply softplus or clamp
    let dt_processed = if dt_softplus {
        client.softplus(&dt_biased).map_err(Error::Numr)?
    } else {
        // Clamp to non-negative
        client
            .clamp(&dt_biased, 0.0, f64::MAX)
            .map_err(Error::Numr)?
    };

    // Reshape to [batch, nchunks, chunk_size, nheads]
    let dt_chunked = dt_processed
        .reshape(&[batch, nchunks, chunk_size, nheads])
        .map_err(Error::Numr)?;

    // Transpose to [batch, nheads, nchunks, chunk_size]
    let dt_out = dt_chunked
        .permute(&[0, 3, 1, 2])
        .map_err(Error::Numr)?
        .contiguous();

    // A: [nheads] → [1, nheads, 1, 1] for broadcast
    let a_broad = a.reshape(&[1, nheads, 1, 1]).map_err(Error::Numr)?;

    // dA = dt_out * A: [batch, nheads, nchunks, chunk_size]
    let dA = client.mul(&dt_out, &a_broad).map_err(Error::Numr)?;

    // Inclusive cumsum along last dim (chunk_size)
    let dA_cumsum = client.cumsum(&dA, 3).map_err(Error::Numr)?;

    Ok((dt_out, dA_cumsum))
}
