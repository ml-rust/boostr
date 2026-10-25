//! Per-chunk intermediates of the chunked gated delta rule.

use super::chunk_layout::{gate_to_chunks, pad_seq, to_chunks};
use super::chunk_mask::{ChunkMasks, chunk_masks};
use super::common::GdnDims;
use super::ut_transform::ut_transform;
use crate::error::{Error, Result};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, CumulativeOps, LinalgOps, MatmulOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Tensors built from chunked inputs, before the sequential loop over chunks.
///
/// `g_cs` is the cumulative gate within each chunk and `N = batch * H`.
/// The chunked backward reads the same tensors. With segment ids,
/// `decay`, `e_g`, `e_tail` and `g_last_exp` carry the document masks of
/// [`ChunkMasks`]. Every term built from them mixes tokens of one
/// document only.
pub(super) struct ChunkTerms<R: Runtime> {
    /// `exp(g_cs[j] - g_cs[i]) * same[j][i]` for `i <= j`, `same` above the
    /// diagonal: `[N, n_chunks, cs, cs]`.
    pub decay: Tensor<R>,
    /// Strictly lower `(k * beta) @ k^T * decay`: `[N, n_chunks, cs, cs]`.
    pub a: Tensor<R>,
    /// Lower `q @ k^T * decay`, diagonal included: `[N, n_chunks, cs, cs]`.
    pub kq: Tensor<R>,
    /// `(I + A)^-1`: `[N, n_chunks, cs, cs]`.
    pub t: Tensor<R>,
    /// `T @ (v * beta)`: `[N, n_chunks, cs, S_v]`.
    pub u: Tensor<R>,
    /// `T @ (k * beta * e_g)`: `[N, n_chunks, cs, S_k]`.
    pub w: Tensor<R>,
    /// `q * e_g`: `[N, n_chunks, cs, S_k]`.
    pub q_g: Tensor<R>,
    /// `(k * e_tail)^T`: `[N, n_chunks, S_k, cs]`.
    pub k_g_t: Tensor<R>,
    /// `exp(g_last) * pre_last` of each chunk: `[N, n_chunks, 1]`.
    pub g_last_exp: Tensor<R>,
    /// `exp(g_cs) * pre`: `[N, n_chunks, cs, 1]`.
    pub e_g: Tensor<R>,
    /// `exp(g_last - g_cs) * last`: `[N, n_chunks, cs, 1]`.
    pub e_tail: Tensor<R>,
}

/// Build the per-chunk intermediates from chunked inputs.
///
/// `q` and `k` are `[N, n_chunks, cs, S_k]`, `v` is `[N, n_chunks, cs, S_v]`,
/// `g` is `[N, n_chunks, cs]` and `beta` is `[N, n_chunks, cs, 1]`.
/// `q` must already carry its `1 / sqrt(S_k)` scale. `masks` is `None` for
/// one document per row: the terms then take no mask factor at all.
pub(super) fn chunk_terms<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    masks: Option<&ChunkMasks<R>>,
) -> Result<ChunkTerms<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ShapeOps<R>
        + MatmulOps<R>
        + CumulativeOps<R>
        + UtilityOps<R>
        + LinalgOps<R>
        + LinearAlgebraAlgorithms<R>
        + TensorOps<R>,
{
    let (n, n_chunks, cs) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let dtype = q.dtype();
    let device = q.device();

    let k_b = client.mul(k, beta)?;
    let v_b = client.mul(v, beta)?;

    let (mask_lower, mask_strict) = tri_masks::<R, C>(client, cs, dtype, device)?;

    // decay[j][i] = exp(g_cs[j] - g_cs[i]) for i <= j. The difference is
    // zeroed above the diagonal before exp (as ggml_tri does) so no
    // positive exponent is ever evaluated. A and KQ mask it out again.
    let g_cs = client.cumsum(g, 2)?; // [N, n_chunks, cs]
    let g_j = g_cs.unsqueeze(3)?; // [N, n_chunks, cs, 1]
    let g_i = g_cs.unsqueeze(2)?; // [N, n_chunks, 1, cs]
    let diff = client.sub(&g_j, &g_i)?;
    let diff = client.mul(&diff, &mask_lower)?;
    let decay = client.exp(&diff)?;
    // Masks multiply in exact 0/1 factors: no cross-document entry survives.
    let decay = match masks {
        Some(m) => client.mul(&decay, &m.same)?,
        None => decay,
    };

    let k_t = k.transpose(2, 3)?;
    let a = client.matmul(&k_b, &k_t)?; // A[j][i] = beta_j k_j . k_i
    let a = client.mul(&client.mul(&a, &decay)?, &mask_strict)?;
    let kq = client.matmul(q, &k_t)?; // KQ[j][i] = q_j . k_i
    let kq = client.mul(&client.mul(&kq, &decay)?, &mask_lower)?;

    let t = ut_transform(client, &a, n, n_chunks, cs)?; // [N, n_chunks, cs, cs]

    let u = client.matmul(&t, &v_b)?; // [N, n_chunks, cs, S_v]
    // The chunk-start state reaches only tokens with pre = 1.
    let e_g = client.exp(&g_cs)?.unsqueeze(3)?; // [N, n_chunks, cs, 1]
    let e_g = match masks {
        Some(m) => client.mul(&e_g, &m.pre)?,
        None => e_g,
    };
    let w = client.matmul(&t, &client.mul(&k_b, &e_g)?)?; // [N, n_chunks, cs, S_k]
    let q_g = client.mul(q, &e_g)?;

    // Only the last document of a chunk writes the carried state, and the
    // old state survives only a chunk that holds one document.
    let g_last = g_cs.narrow(2, cs - 1, 1)?.contiguous()?; // [N, n_chunks, 1]
    let g_last_exp = client.exp(&g_last)?;
    let e_tail = client.exp(&client.sub(&g_last, &g_cs)?)?.unsqueeze(3)?;
    let (e_tail, g_last_exp) = match masks {
        Some(m) => (
            client.mul(&e_tail, &m.last)?,
            client.mul(&g_last_exp, &m.pre_last)?,
        ),
        None => (e_tail, g_last_exp),
    };
    let k_g_t = client.mul(k, &e_tail)?.transpose(2, 3)?; // [N, n_chunks, S_k, cs]

    Ok(ChunkTerms {
        decay,
        a,
        kq,
        t,
        u,
        w,
        q_g,
        k_g_t,
        g_last_exp,
        e_g,
        e_tail,
    })
}

/// Lower (diagonal included) and strictly lower masks: `[1, 1, cs, cs]`.
///
/// Both broadcast over `(N, n_chunks)`.
pub(super) fn tri_masks<R, C>(
    client: &C,
    cs: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: LinalgOps<R>,
{
    let ones = Tensor::<R>::ones(&[cs, cs], dtype, device)?;
    let lower = LinalgOps::tril(client, &ones, 0)?.reshape(&[1, 1, cs, cs])?;
    let strict = LinalgOps::tril(client, &ones, -1)?.reshape(&[1, 1, cs, cs])?;
    Ok((lower, strict))
}

/// Inputs padded to a chunk multiple, split into chunks, with their terms.
pub(super) struct Chunked<R: Runtime> {
    /// `batch * H`.
    pub n: usize,
    /// Chunks per sequence.
    pub n_chunks: usize,
    /// `n_chunks * chunk_size`.
    pub padded: usize,
    /// `q / sqrt(S_k)`: `[N, n_chunks, cs, S_k]`.
    pub q: Tensor<R>,
    /// `[N, n_chunks, cs, S_k]`.
    pub k: Tensor<R>,
    /// `[N, n_chunks, cs, S_v]`.
    pub v: Tensor<R>,
    /// `[N, n_chunks, cs, 1]`.
    pub beta: Tensor<R>,
    /// Intermediates built from the chunked inputs.
    pub terms: ChunkTerms<R>,
}

/// Pad and split the inputs into chunks, then build their terms.
///
/// Padded tokens carry `beta = 0` and `g = 0`. Their rows of `A`, `U` and
/// `W` vanish, so the state never sees them. `segment_ids` is an I32
/// `[batch, seq]` tensor or `None` for one document per row.
#[allow(
    clippy::too_many_arguments,
    reason = "client, five forward inputs, segment ids, the dims and the chunk size"
)]
pub(super) fn chunk_inputs<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    segment_ids: Option<&Tensor<R>>,
    dims: GdnDims,
    chunk_size: usize,
) -> Result<Chunked<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + MatmulOps<R>
        + CumulativeOps<R>
        + UtilityOps<R>
        + LinalgOps<R>
        + LinearAlgebraAlgorithms<R>
        + TensorOps<R>
        + CompareOps<R>
        + TypeConversionOps<R>,
{
    if chunk_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "chunk_size",
            reason: "must be >= 1".into(),
        });
    }
    let (seq, cs) = (dims.seq, chunk_size);
    let n = dims.batch * dims.heads;
    let n_chunks = seq.div_ceil(cs);
    let padded = n_chunks * cs;

    let q = to_chunks(&pad_seq(client, q, seq, padded)?, n, n_chunks, cs)?;
    let q = client.mul_scalar(&q, 1.0 / (dims.s_k as f64).sqrt())?;
    let k = to_chunks(&pad_seq(client, k, seq, padded)?, n, n_chunks, cs)?;
    let v = to_chunks(&pad_seq(client, v, seq, padded)?, n, n_chunks, cs)?;
    let g = gate_to_chunks(&pad_seq(client, g, seq, padded)?, n, n_chunks, cs)?;
    let beta = gate_to_chunks(&pad_seq(client, beta, seq, padded)?, n, n_chunks, cs)?
        .reshape(&[n, n_chunks, cs, 1])?;
    let masks = segment_ids
        .map(|ids| chunk_masks(client, ids, dims, n_chunks, cs, q.dtype()))
        .transpose()?;
    let terms = chunk_terms(client, &q, &k, &v, &g, &beta, masks.as_ref())?;
    Ok(Chunked {
        n,
        n_chunks,
        padded,
        q,
        k,
        v,
        beta,
        terms,
    })
}

/// One chunk of the state recurrence.
///
/// ```text
/// V_new = U - W @ S
/// S'    = S * exp(g_last) + K_g^T @ V_new
/// ```
///
/// Operands are one chunk: `s: [N, S_k, S_v]`, `w_c: [N, cs, S_k]`,
/// `u_c: [N, cs, S_v]`, `k_g_t_c: [N, S_k, cs]`, `g_last_exp_c: [N, 1, 1]`.
/// Returns `(V_new, S')`.
pub(super) fn advance_state<R, C>(
    client: &C,
    s: &Tensor<R>,
    w_c: &Tensor<R>,
    u_c: &Tensor<R>,
    k_g_t_c: &Tensor<R>,
    g_last_exp_c: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + MatmulOps<R>,
{
    let v_new = client.sub(u_c, &client.matmul(w_c, s)?)?; // [N, cs, S_v]
    let ds = client.matmul(k_g_t_c, &v_new)?; // [N, S_k, S_v]
    let s_next = client.add(&client.mul(s, g_last_exp_c)?, &ds)?;
    Ok((v_new, s_next))
}
