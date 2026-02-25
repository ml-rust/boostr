//! Generic SSD kernel implementations
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives: exp, cumsum, matmul, mul, add, narrow, reshape.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CumulativeOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    UnaryOps, UtilityOps,
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

/// Compute per-chunk final hidden states.
///
/// For each chunk c, position j:
///   h_c = Σ_j exp(dA_last - dA_j) * dt_j * B_j ⊗ x_j
///
/// Returns `states: [batch, nchunks, nheads, headdim, dstate]`
#[allow(non_snake_case)]
pub fn ssd_chunk_state_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    b: &Tensor<R>,
    dt: &Tensor<R>,
    dA_cumsum: &Tensor<R>,
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
    // b: [batch, seqlen, ngroups, dstate]
    // dt: [batch, nheads, nchunks, chunk_size]
    // dA_cumsum: [batch, nheads, nchunks, chunk_size]
    let x_shape = x.shape();
    let b_shape = b.shape();
    let da_shape = dA_cumsum.shape();

    let batch = x_shape[0];
    let seqlen = x_shape[1];
    let nheads = x_shape[2];
    let headdim = x_shape[3];
    let ngroups = b_shape[2];
    let dstate = b_shape[3];
    let nchunks = da_shape[2];
    let chunk_size = da_shape[3];

    let device = x.device();
    let dtype = x.dtype();

    // Pad x and b to nchunks * chunk_size if needed
    let padded_len = nchunks * chunk_size;
    let x_padded = if padded_len > seqlen {
        let pad = Tensor::<R>::zeros(
            &[batch, padded_len - seqlen, nheads, headdim],
            dtype,
            device,
        );
        client.cat(&[x, &pad], 1).map_err(Error::Numr)?
    } else {
        x.clone()
    };
    let b_padded = if padded_len > seqlen {
        let pad = Tensor::<R>::zeros(
            &[batch, padded_len - seqlen, ngroups, dstate],
            dtype,
            device,
        );
        client.cat(&[b, &pad], 1).map_err(Error::Numr)?
    } else {
        b.clone()
    };

    // dA_last: last element of each chunk = dA_cumsum[:, :, :, -1]
    // Shape: [batch, nheads, nchunks, 1]
    let dA_last = dA_cumsum
        .narrow(3, chunk_size - 1, 1)
        .map_err(Error::Numr)?;

    // decay = exp(dA_last - dA_cumsum): [batch, nheads, nchunks, chunk_size]
    let dA_diff = client.sub(&dA_last, dA_cumsum).map_err(Error::Numr)?;
    // Clamp to <= 0 for numerical stability
    let dA_clamped = client
        .clamp(&dA_diff, f64::NEG_INFINITY, 0.0)
        .map_err(Error::Numr)?;
    let decay = client.exp(&dA_clamped).map_err(Error::Numr)?;

    // scale = decay * dt: [batch, nheads, nchunks, chunk_size]
    let scale = client.mul(&decay, dt).map_err(Error::Numr)?;

    // Reshape x to chunks: [batch, nchunks, chunk_size, nheads, headdim]
    let x_chunks = x_padded
        .reshape(&[batch, nchunks, chunk_size, nheads, headdim])
        .map_err(Error::Numr)?;

    // scale: [batch, nheads, nchunks, chunk_size] → [batch, nchunks, chunk_size, nheads, 1]
    let scale_t = scale
        .permute(&[0, 2, 3, 1])
        .map_err(Error::Numr)?
        .contiguous()
        .reshape(&[batch, nchunks, chunk_size, nheads, 1])
        .map_err(Error::Numr)?;

    // scaled_x = x_chunks * scale: [batch, nchunks, chunk_size, nheads, headdim]
    let scaled_x = client.mul(&x_chunks, &scale_t).map_err(Error::Numr)?;

    // Reshape B to chunks: [batch, nchunks, chunk_size, ngroups, dstate]
    let b_chunks = b_padded
        .reshape(&[batch, nchunks, chunk_size, ngroups, dstate])
        .map_err(Error::Numr)?;

    // For state computation: scaled_x^T @ B per chunk
    // scaled_x: [batch, nchunks, chunk_size, nheads, headdim]
    // b_chunks: [batch, nchunks, chunk_size, ngroups, dstate]
    // We need: Σ_j scaled_x[j] ⊗ B[j] = scaled_x^T @ B
    //
    // Approach: for each head, sum over chunk positions.
    // With ngroups == 1 or ngroups == nheads:
    //   states[b,c,h,d,n] = Σ_j scaled_x[b,c,j,h,d] * B[b,c,j,g,n]
    //
    // Use matmul: reshape scaled_x to [..., headdim, chunk_size] and
    // B to [..., chunk_size, dstate], then matmul.
    let heads_per_group = nheads / ngroups;

    // Transpose scaled_x: [batch, nchunks, nheads, headdim, chunk_size]
    let sx_t = scaled_x
        .permute(&[0, 1, 3, 4, 2])
        .map_err(Error::Numr)?
        .contiguous();

    // Expand B for head groups: [batch, nchunks, nheads, chunk_size, dstate]
    let b_for_heads = if ngroups == nheads {
        b_chunks
            .permute(&[0, 1, 3, 4, 2])
            .map_err(Error::Numr)?
            .contiguous()
            .permute(&[0, 1, 2, 4, 3])
            .map_err(Error::Numr)?
            .contiguous()
    } else {
        // ngroups < nheads: repeat each group heads_per_group times
        // [batch, nchunks, chunk_size, ngroups, dstate]
        //   → [batch, nchunks, ngroups, chunk_size, dstate]
        //   → repeat → [batch, nchunks, nheads, chunk_size, dstate]
        let b_t = b_chunks
            .permute(&[0, 1, 3, 2, 4])
            .map_err(Error::Numr)?
            .contiguous();
        if heads_per_group == 1 {
            b_t
        } else {
            // Repeat along group dim
            b_t.reshape(&[batch, nchunks, ngroups, 1, chunk_size, dstate])
                .map_err(Error::Numr)?
                .broadcast_to(&[batch, nchunks, ngroups, heads_per_group, chunk_size, dstate])
                .map_err(Error::Numr)?
                .contiguous()
                .reshape(&[batch, nchunks, nheads, chunk_size, dstate])
                .map_err(Error::Numr)?
        }
    };

    // matmul: [batch, nchunks, nheads, headdim, chunk_size]
    //       @ [batch, nchunks, nheads, chunk_size, dstate]
    //       → [batch, nchunks, nheads, headdim, dstate]
    let states = client.matmul(&sx_t, &b_for_heads).map_err(Error::Numr)?;

    Ok(states)
}

/// Propagate states across chunks: h[c] = exp(dA_last[c]) * h[c-1] + h[c].
///
/// Returns `states_out: [batch, nchunks, nheads, headdim, dstate]`
#[allow(non_snake_case)]
pub fn ssd_state_passing_impl<R, C>(
    client: &C,
    states: &Tensor<R>,
    dA_cumsum: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ShapeOps<R>
        + ScalarOps<R>
        + UtilityOps<R>
        + TensorOps<R>,
{
    // states: [batch, nchunks, nheads, headdim, dstate]
    // dA_cumsum: [batch, nheads, nchunks, chunk_size]
    let s_shape = states.shape();
    let da_shape = dA_cumsum.shape();

    let batch = s_shape[0];
    let nchunks = s_shape[1];
    let nheads = s_shape[2];
    let _headdim = s_shape[3];
    let _dstate = s_shape[4];
    let chunk_size = da_shape[3];

    if nchunks <= 1 {
        return Ok(states.clone());
    }

    // dA_last: dA_cumsum[:, :, :, -1] → [batch, nheads, nchunks]
    let dA_last = dA_cumsum
        .narrow(3, chunk_size - 1, 1)
        .map_err(Error::Numr)?
        .contiguous()
        .reshape(&[batch, nheads, nchunks])
        .map_err(Error::Numr)?;

    // Clamp and exponentiate: [batch, nheads, nchunks]
    let dA_clamped = client
        .clamp(&dA_last, f64::NEG_INFINITY, 0.0)
        .map_err(Error::Numr)?;
    let dA_scale = client.exp(&dA_clamped).map_err(Error::Numr)?;

    // Sequential scan: for c in 1..nchunks:
    //   states[:, c] += dA_scale[:, :, c] * states[:, c-1]
    //
    // `to_vec` here is called only by CPU backend — CUDA and WebGPU backends
    // use fused kernels. On CPU there is no device memory.
    //
    // For the impl_generic path we accumulate chunk-by-chunk using narrow+add.
    // This is O(nchunks) kernel launches but nchunks is typically small (L/chunk_size).

    // Collect chunks
    let mut chunks: Vec<Tensor<R>> = Vec::with_capacity(nchunks);
    let first = states.narrow(1, 0, 1).map_err(Error::Numr)?;
    chunks.push(first);

    for c in 1..nchunks {
        let prev = &chunks[c - 1]; // [batch, 1, nheads, headdim, dstate]

        // decay for this chunk: dA_scale[:, :, c] → [batch, 1, nheads, 1, 1]
        let scale_c = dA_scale
            .narrow(2, c, 1)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[batch, 1, nheads, 1, 1])
            .map_err(Error::Numr)?;

        // decayed_prev = scale * prev: [batch, 1, nheads, headdim, dstate]
        let decayed = client.mul(&scale_c, prev).map_err(Error::Numr)?;

        // current chunk
        let curr = states.narrow(1, c, 1).map_err(Error::Numr)?;

        // new_c = curr + decayed
        let new_c = client.add(&curr, &decayed).map_err(Error::Numr)?;
        chunks.push(new_c);
    }

    // Concatenate along dim 1
    let chunk_refs: Vec<&Tensor<R>> = chunks.iter().collect();
    client.cat(&chunk_refs, 1).map_err(Error::Numr)
}

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
    // y = C @ states^T = Σ_n C[..., n] * states[..., d, n] over n
    // → y: [batch, nchunks, chunk_size, nheads, headdim]
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
