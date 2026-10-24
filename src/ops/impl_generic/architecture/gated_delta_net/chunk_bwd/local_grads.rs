//! Chunk-local stage of the chunked GDN backward.
//!
//! With the chunk-start state `S` and the chunk-end adjoint `dS'` known,
//! every chunk is independent. Each product below runs once over
//! `[N, n_chunks, ..]`. Names follow `ChunkTerms`: `G = g_cs`,
//! `gamma = G[last]`, `Q~ = q / sqrt(S_k)`, `K_b = beta K`, `V_b = beta V`,
//! `K_be = K_b e_g`, `P = kq`, `D = decay`.
//!
//! The exponential factors come from `ChunkTerms` and are never rebuilt
//! here: `e_g = exp(G) pre` and `e_tail = exp(gamma - G) last`. `D` and
//! `exp(gamma)` carry their document masks too. The masks are constants,
//! so every product below keeps them and no gradient crosses a document
//! start. Below, `exp(G)` stands for `e_g` and `exp(gamma - G)` for `e_tail`.
//!
//! ```text
//! dV_new = P^T dO + K_g dS'        dQ_g = dO S^T
//! dP     = lower(dO V_new^T)       dK_g = V_new dS'^T
//! dW     = -dV_new S^T             dT   = dV_new V_b^T + dW K_be^T
//! dV_b   = T^T dV_new              dK_be = T^T dW
//! dA     = strict(-T^T dT T^T)
//! dK_b   = (dA D) K + dK_be exp(G)
//! dK     = (dA D)^T K_b + (dP D)^T Q~ + dK_g exp(gamma - G) + beta dK_b
//! dQ~    = (dP D) K + dQ_g exp(G)
//! dG     = decay terms + rowsum(dQ_g Q_g) - rowsum(dK_g K_g) + rowsum(dK_be K_be)
//! ```

use super::super::chunk_terms::{Chunked, tri_masks};
use super::gates::{beta_grads, decay_grad, gate_grad, last_gate_grad, row_dot};
use super::states::ChunkStates;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CumulativeOps, LinalgOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Chunked input gradients. Padded tokens are still present.
pub(super) struct LocalGrads<R: Runtime> {
    /// `dL/dq` for the unscaled `q`: `[N, n_chunks, cs, S_k]`.
    pub dq: Tensor<R>,
    /// `[N, n_chunks, cs, S_k]`.
    pub dk: Tensor<R>,
    /// `[N, n_chunks, cs, S_v]`.
    pub dv: Tensor<R>,
    /// `[N, n_chunks, cs]`.
    pub dg: Tensor<R>,
    /// `[N, n_chunks, cs]`.
    pub dbeta: Tensor<R>,
}

/// Input gradients of every chunk, batched over all chunks.
///
/// `ds_end` is `[N, n_chunks, S_k, S_v]`. `d_o` is `[N, n_chunks, cs, S_v]`,
/// zero on padded tokens. `scale` is the `1 / sqrt(S_k)` the forward puts
/// on `q`.
pub(super) fn local_grads<R, C>(
    client: &C,
    x: &Chunked<R>,
    states: &ChunkStates<R>,
    ds_end: &Tensor<R>,
    d_o: &Tensor<R>,
    scale: f64,
) -> Result<LocalGrads<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + MatmulOps<R>
        + ReduceOps<R>
        + CumulativeOps<R>
        + LinalgOps<R>,
{
    let t = &x.terms;
    let cs = x.k.shape()[2];
    let (mask_lower, mask_strict) = tri_masks::<R, C>(client, cs, x.k.dtype(), x.k.device())?;
    let s = &states.s;
    let s_t = s.transpose(2, 3)?;
    let v_new = &states.v_new;

    let (e_g, e_tail) = (&t.e_g, &t.e_tail); // [N, n_chunks, cs, 1]
    let k_g = t.k_g_t.transpose(2, 3)?; // [N, n_chunks, cs, S_k]

    // Output and state paths.
    let dv_new = client.add(
        &client.matmul(&t.kq.transpose(2, 3)?, d_o)?,
        &client.matmul(&k_g, ds_end)?,
    )?;
    let dqg = client.matmul(d_o, &s_t)?;
    let dp = client.mul(&client.matmul(d_o, &v_new.transpose(2, 3)?)?, &mask_lower)?;
    let dkg = client.matmul(v_new, &ds_end.transpose(2, 3)?)?;
    let dw = client.neg(&client.matmul(&dv_new, &s_t)?)?;

    // UT transform: U = T V_b, W = T K_be, T = (I + A)^-1.
    let k_b = client.mul(&x.k, &x.beta)?;
    let v_b = client.mul(&x.v, &x.beta)?;
    let kbe = client.mul(&k_b, e_g)?;
    let dt = client.add(
        &client.matmul(&dv_new, &v_b.transpose(2, 3)?)?,
        &client.matmul(&dw, &kbe.transpose(2, 3)?)?,
    )?;
    let t_t = t.t.transpose(2, 3)?;
    let dv_b = client.matmul(&t_t, &dv_new)?;
    let dkbe = client.matmul(&t_t, &dw)?;
    let da = client.matmul(&client.matmul(&t_t, &dt)?, &t_t)?;
    let da = client.mul(&client.neg(&da)?, &mask_strict)?;

    // Through A = strict(K_b K^T D) and P = lower(Q~ K^T D).
    let dm_a = client.mul(&da, &t.decay)?;
    let dm_p = client.mul(&dp, &t.decay)?;
    let dk_b = client.add(&client.matmul(&dm_a, &x.k)?, &client.mul(&dkbe, e_g)?)?;
    let dk = client.add(
        &client.matmul(&dm_a.transpose(2, 3)?, &k_b)?,
        &client.matmul(&dm_p.transpose(2, 3)?, &x.q)?,
    )?;
    let dk = client.add(&dk, &client.mul(&dkg, e_tail)?)?;
    let dq = client.add(&client.matmul(&dm_p, &x.k)?, &client.mul(&dqg, e_g)?)?;
    let dq = client.mul_scalar(&dq, scale)?;

    // Gates.
    let d_gamma = decay_grad(client, &da, &t.a, &dp, &t.kq)?;
    let d_gamma = client.add(&d_gamma, &row_dot(client, &dqg, &t.q_g)?)?;
    let d_gamma = client.sub(&d_gamma, &row_dot(client, &dkg, &k_g)?)?;
    let d_gamma = client.add(&d_gamma, &row_dot(client, &dkbe, &kbe)?)?;
    let d_last = last_gate_grad(client, &dkg, &k_g, ds_end, s, &t.g_last_exp)?;
    let dg = gate_grad(client, &d_gamma, &d_last)?;

    let b = beta_grads(client, &x.k, &x.v, &x.beta, &dk_b, &dv_b)?;
    Ok(LocalGrads {
        dq,
        dk: client.add(&dk, &b.dk)?,
        dv: b.dv,
        dg,
        dbeta: b.dbeta,
    })
}
