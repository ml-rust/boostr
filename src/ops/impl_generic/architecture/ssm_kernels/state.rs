//! SSD chunk state and state-passing kernels

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
        let b_t = b_chunks
            .permute(&[0, 1, 3, 2, 4])
            .map_err(Error::Numr)?
            .contiguous();
        if heads_per_group == 1 {
            b_t
        } else {
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
