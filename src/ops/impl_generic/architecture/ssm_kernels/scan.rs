//! SSD chunk scan kernel

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Chunk scan: y[t] = C[t] @ h[chunk_of(t)] + D * x[t].
///
/// This computes the off-diagonal contribution from propagated states.
/// The full intra-chunk diagonal contribution (L ⊙ (C @ B^T) @ X) is
/// handled by the caller; this function covers the inter-chunk state
/// projection.
///
/// Returns `output: [batch, seqlen, nheads, headdim]`
#[allow(non_snake_case)]
pub fn ssd_chunk_scan_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    states: &Tensor<R>,
    c: &Tensor<R>,
    dA_cumsum: &Tensor<R>,
    d: Option<&Tensor<R>>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ShapeOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + TensorOps<R>,
{
    // x: [batch, seqlen, nheads, headdim]
    // states: [batch, nchunks, nheads, headdim, dstate]
    // c: [batch, seqlen, ngroups, dstate]
    // dA_cumsum: [batch, nheads, nchunks, chunk_size]
    // d: optional [nheads]
    let x_shape = x.shape();
    let s_shape = states.shape();
    let c_shape = c.shape();
    let da_shape = dA_cumsum.shape();

    let batch = x_shape[0];
    let seqlen = x_shape[1];
    let nheads = x_shape[2];
    let headdim = x_shape[3];
    let nchunks = s_shape[1];
    let dstate = s_shape[4];
    let ngroups = c_shape[2];
    let chunk_size = da_shape[3];
    let heads_per_group = nheads / ngroups;

    let device = x.device();
    let dtype = x.dtype();
    let padded_len = nchunks * chunk_size;

    // Pad C to padded_len
    let c_padded = if padded_len > seqlen {
        let pad = Tensor::<R>::zeros(
            &[batch, padded_len - seqlen, ngroups, dstate],
            dtype,
            device,
        );
        client.cat(&[c, &pad], 1).map_err(Error::Numr)?
    } else {
        c.clone()
    };

    // Decay from chunk start: exp(dA_cumsum) for intra-chunk position weighting
    // dA_cumsum: [batch, nheads, nchunks, chunk_size]
    let dA_clamped = client
        .clamp(dA_cumsum, f64::NEG_INFINITY, 0.0)
        .map_err(Error::Numr)?;
    let decay = client.exp(&dA_clamped).map_err(Error::Numr)?;

    // Reshape C to chunks: [batch, nchunks, chunk_size, ngroups, dstate]
    let c_chunks = c_padded
        .reshape(&[batch, nchunks, chunk_size, ngroups, dstate])
        .map_err(Error::Numr)?;

    // Expand C for heads: [batch, nchunks, chunk_size, nheads, dstate]
    let c_for_heads = if ngroups == nheads {
        c_chunks.permute(&[0, 1, 2, 3, 4]).map_err(Error::Numr)?
    } else if heads_per_group == 1 {
        c_chunks
    } else {
        c_chunks
            .reshape(&[batch, nchunks, chunk_size, ngroups, 1, dstate])
            .map_err(Error::Numr)?
            .broadcast_to(&[batch, nchunks, chunk_size, ngroups, heads_per_group, dstate])
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[batch, nchunks, chunk_size, nheads, dstate])
            .map_err(Error::Numr)?
    };

    // states: [batch, nchunks, nheads, headdim, dstate]
    // C: [batch, nchunks, chunk_size, nheads, dstate]
    // y = C @ states^T → [batch, nchunks, chunk_size, nheads, headdim]
    //
    // Reshape for batched matmul:
    // C: [batch * nchunks * nheads, chunk_size, dstate]
    // states: [batch * nchunks * nheads, headdim, dstate]
    // y = C @ states^T → [batch * nchunks * nheads, chunk_size, headdim]

    // Transpose C: [batch, nchunks, nheads, chunk_size, dstate]
    let c_t = c_for_heads
        .permute(&[0, 1, 3, 2, 4])
        .map_err(Error::Numr)?
        .contiguous()
        .reshape(&[batch * nchunks * nheads, chunk_size, dstate])
        .map_err(Error::Numr)?;

    // states^T: [batch * nchunks * nheads, dstate, headdim]
    let states_t = states
        .reshape(&[batch * nchunks * nheads, headdim, dstate])
        .map_err(Error::Numr)?
        .permute(&[0, 2, 1])
        .map_err(Error::Numr)?
        .contiguous();

    // matmul: [B*K*H, chunk_size, dstate] @ [B*K*H, dstate, headdim]
    //       → [B*K*H, chunk_size, headdim]
    let y_flat = client.matmul(&c_t, &states_t).map_err(Error::Numr)?;

    // Reshape back: [batch, nchunks, nheads, chunk_size, headdim]
    let y_chunked = y_flat
        .reshape(&[batch, nchunks, nheads, chunk_size, headdim])
        .map_err(Error::Numr)?;

    // Apply intra-chunk decay: [batch, nheads, nchunks, chunk_size]
    //   → [batch, nchunks, nheads, chunk_size, 1]
    let decay_t = decay
        .permute(&[0, 2, 1, 3])
        .map_err(Error::Numr)?
        .contiguous()
        .reshape(&[batch, nchunks, nheads, chunk_size, 1])
        .map_err(Error::Numr)?;

    let y_decayed = client.mul(&y_chunked, &decay_t).map_err(Error::Numr)?;

    // Reshape to [batch, nchunks * chunk_size, nheads, headdim]
    let y_full = y_decayed
        .permute(&[0, 1, 3, 2, 4])
        .map_err(Error::Numr)?
        .contiguous()
        .reshape(&[batch, padded_len, nheads, headdim])
        .map_err(Error::Numr)?;

    // Trim to original seqlen
    let y_trimmed = if padded_len > seqlen {
        y_full.narrow(1, 0, seqlen).map_err(Error::Numr)?
    } else {
        y_full
    };

    // D skip connection: y += D * x
    let output = if let Some(d_param) = d {
        let d_broad = d_param.reshape(&[1, 1, nheads, 1]).map_err(Error::Numr)?;
        let d_x = client.mul(&d_broad, x).map_err(Error::Numr)?;
        client.add(&y_trimmed, &d_x).map_err(Error::Numr)?
    } else {
        y_trimmed
    };

    Ok(output)
}
