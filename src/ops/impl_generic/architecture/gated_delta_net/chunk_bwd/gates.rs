//! Gate gradients of the chunked GDN backward.
//!
//! `G = cumsum(g)` within a chunk and `gamma = G[last]`. The local stage
//! collects `dG` and `dgamma`. This file maps them to `dg`, and maps the
//! `beta`-scaled operand gradients to `dk`, `dv` and `dbeta`.

use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CumulativeOps, ReduceOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `sum(x * y)` over the last axis: `[N, n_chunks, cs, S] -> [N, n_chunks, cs]`.
pub(super) fn row_dot<R, C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ReduceOps<R>,
{
    Ok(client.sum(&client.mul(x, y)?, &[3], false)?)
}

/// `dG` through the decay matrix `D[j, i] = exp(G[j] - G[i])`.
///
/// `A = M_A * D` and `P = M_P * D` on their masks, so
/// `dD * D = dA * A + dP * P`. With `E = dA * A + dP * P`:
///
/// ```text
/// dG[j] += sum_i E[j, i]
/// dG[i] -= sum_j E[j, i]
/// ```
///
/// `da` and `dp` must be zero off their masks. All operands are
/// `[N, n_chunks, cs, cs]`. Returns `[N, n_chunks, cs]`.
pub(super) fn decay_grad<R, C>(
    client: &C,
    da: &Tensor<R>,
    a: &Tensor<R>,
    dp: &Tensor<R>,
    p: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ReduceOps<R>,
{
    let e = client.add(&client.mul(da, a)?, &client.mul(dp, p)?)?;
    let rows = client.sum(&e, &[3], false)?;
    let cols = client.sum(&e, &[2], false)?;
    Ok(client.sub(&rows, &cols)?)
}

/// `dgamma = sum(dK_g * K_g) + exp(gamma) <dS', S>`: `[N, n_chunks, 1]`.
///
/// `dkg` and `k_g` are `[N, n_chunks, cs, S_k]`. `ds_end` and `s` are
/// `[N, n_chunks, S_k, S_v]`. `g_last_exp` is `[N, n_chunks, 1]`.
pub(super) fn last_gate_grad<R, C>(
    client: &C,
    dkg: &Tensor<R>,
    k_g: &Tensor<R>,
    ds_end: &Tensor<R>,
    s: &Tensor<R>,
    g_last_exp: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ReduceOps<R> + ShapeOps<R>,
{
    let through_kg = client.sum(&client.mul(dkg, k_g)?, &[2, 3], false)?;
    let through_s = client.sum(&client.mul(ds_end, s)?, &[2, 3], false)?;
    let through_s = client.mul(&through_s.unsqueeze(2)?, g_last_exp)?;
    Ok(client.add(&through_kg.unsqueeze(2)?, &through_s)?)
}

/// `dg` from `dG` and `dgamma`.
///
/// `dg[t] = sum_{s >= t} dG[s] + dgamma`, since `gamma` sums every gate of
/// the chunk. The suffix sum is `total - cumsum + dG`. `d_gamma` is
/// `[N, n_chunks, cs]` and `d_last` is `[N, n_chunks, 1]`.
pub(super) fn gate_grad<R, C>(
    client: &C,
    d_gamma: &Tensor<R>,
    d_last: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ReduceOps<R> + CumulativeOps<R>,
{
    let total = client.sum(d_gamma, &[2], true)?;
    let prefix = client.cumsum(d_gamma, 2)?;
    let suffix = client.add(&client.sub(&total, &prefix)?, d_gamma)?;
    Ok(client.add(&suffix, d_last)?)
}

/// Gradients through `K_b = beta K` and `V_b = beta V`.
pub(super) struct BetaGrads<R: Runtime> {
    /// `beta dK_b`, to add to `dk`: `[N, n_chunks, cs, S_k]`.
    pub dk: Tensor<R>,
    /// `beta dV_b`: `[N, n_chunks, cs, S_v]`.
    pub dv: Tensor<R>,
    /// `sum(dK_b * K) + sum(dV_b * V)` per token: `[N, n_chunks, cs]`.
    pub dbeta: Tensor<R>,
}

/// Map `dK_b` and `dV_b` to `k`, `v` and `beta`. `beta` is `[N, n_chunks, cs, 1]`.
pub(super) fn beta_grads<R, C>(
    client: &C,
    k: &Tensor<R>,
    v: &Tensor<R>,
    beta: &Tensor<R>,
    dk_b: &Tensor<R>,
    dv_b: &Tensor<R>,
) -> Result<BetaGrads<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ReduceOps<R>,
{
    let dbeta = client.add(&row_dot(client, dk_b, k)?, &row_dot(client, dv_b, v)?)?;
    Ok(BetaGrads {
        dk: client.mul(dk_b, beta)?,
        dv: client.mul(dv_b, beta)?,
        dbeta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn gate_grad_is_suffix_sum_plus_last() {
        let (client, device) = cpu_setup();
        let d_gamma =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[1, 1, 3], &device).unwrap();
        let d_last = Tensor::<CpuRuntime>::from_slice(&[10.0f64], &[1, 1, 1], &device).unwrap();
        let dg = gate_grad(&client, &d_gamma, &d_last).unwrap();
        assert_eq!(dg.to_vec::<f64>(), vec![16.0, 15.0, 13.0]);
    }

    /// `E = [[5, 0], [13, 11]]`: rows `[5, 24]`, columns `[18, 11]`.
    #[test]
    fn decay_grad_is_rows_minus_columns() {
        let (client, device) = cpu_setup();
        let mk = |d: &[f64]| Tensor::<CpuRuntime>::from_slice(d, &[1, 1, 2, 2], &device).unwrap();
        let da = mk(&[0.0, 0.0, 2.0, 0.0]);
        let a = mk(&[0.0, 0.0, 3.0, 0.0]);
        let dp = mk(&[1.0, 0.0, 1.0, 1.0]);
        let p = mk(&[5.0, 0.0, 7.0, 11.0]);
        let d = decay_grad(&client, &da, &a, &dp, &p).unwrap();
        assert_eq!(d.to_vec::<f64>(), vec![-13.0, 13.0]);
    }
}
