//! Chunk layout helpers of the chunked gated delta rule.

use super::common::GdnDims;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Zero-pad `t` along `dim` 1 from `seq` to `padded` tokens.
pub(super) fn pad_seq<R, C>(
    client: &C,
    t: &Tensor<R>,
    seq: usize,
    padded: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R>,
{
    if padded == seq {
        return Ok(t.clone());
    }
    let mut pad_shape = t.shape().to_vec();
    pad_shape[1] = padded - seq;
    let pad = Tensor::<R>::zeros(&pad_shape, t.dtype(), t.device())?;
    Ok(client.cat(&[t, &pad], 1)?)
}

/// `[batch, seq, H, S]` -> `[batch * H, n_chunks, chunk, S]`.
pub(super) fn to_chunks<R: Runtime>(
    t: &Tensor<R>,
    n: usize,
    n_chunks: usize,
    chunk: usize,
) -> Result<Tensor<R>> {
    let s = t.shape()[3];
    Ok(t.permute(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[n, n_chunks, chunk, s])?)
}

/// `[batch, seq, H]` -> `[batch * H, n_chunks, chunk]`.
pub(super) fn gate_to_chunks<R: Runtime>(
    t: &Tensor<R>,
    n: usize,
    n_chunks: usize,
    chunk: usize,
) -> Result<Tensor<R>> {
    Ok(t.permute(&[0, 2, 1])?
        .contiguous()?
        .reshape(&[n, n_chunks, chunk])?)
}

/// `[batch * H, .., S]` over `padded` tokens -> `[batch, seq, H, S]`.
///
/// The input is `[N, padded, S]` or `[N, n_chunks, chunk, S]`. Padded
/// tokens are dropped.
pub(super) fn from_chunks<R: Runtime>(
    t: &Tensor<R>,
    dims: GdnDims,
    padded: usize,
) -> Result<Tensor<R>> {
    let s = t.shape()[t.ndim() - 1];
    Ok(t.contiguous()?
        .reshape(&[dims.batch, dims.heads, padded, s])?
        .narrow(2, 0, dims.seq)?
        .permute(&[0, 2, 1, 3])?
        .contiguous()?)
}

/// `[batch * H, n_chunks, chunk]` -> `[batch, seq, H]`, padded tokens dropped.
pub(super) fn gate_from_chunks<R: Runtime>(
    t: &Tensor<R>,
    dims: GdnDims,
    padded: usize,
) -> Result<Tensor<R>> {
    Ok(t.contiguous()?
        .reshape(&[dims.batch, dims.heads, padded])?
        .narrow(2, 0, dims.seq)?
        .permute(&[0, 2, 1])?
        .contiguous()?)
}

/// Chunk `c` of a `[N, n_chunks, ..]` tensor as a contiguous `[N, ..]` tensor.
pub(super) fn chunk_of<R: Runtime>(t: &Tensor<R>, c: usize) -> Result<Tensor<R>> {
    let mut shape = t.shape().to_vec();
    shape.remove(1);
    Ok(t.narrow(1, c, 1)?.contiguous()?.reshape(&shape)?)
}
