//! Forward replay for the sequential GDN backward.
//!
//! Runs the per-token recurrence of
//! [`gdn_step_impl`](super::step::gdn_step_impl) and keeps the tensors of
//! every token that the backward reads.

use super::common::GdnDims;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Forward tensors of one token.
///
/// Vectors are rows: `[batch, H, 1, S]`. Gates are `[batch, H, 1, 1]`.
pub(super) struct Saved<R: Runtime> {
    /// `k_t`: `[batch, H, 1, S_k]`.
    pub k: Tensor<R>,
    /// `beta_t`.
    pub beta: Tensor<R>,
    /// `a = exp(g_t)`.
    pub decay: Tensor<R>,
    /// `Sbar = a * S_{t-1}`: `[batch, H, S_k, S_v]`.
    pub sbar: Tensor<R>,
    /// `v - p` with `p = k @ Sbar`: `[batch, H, 1, S_v]`.
    pub resid: Tensor<R>,
    /// `d = beta * (v - p)`: `[batch, H, 1, S_v]`.
    pub d: Tensor<R>,
    /// `S_t = Sbar + k^T @ d`: `[batch, H, S_k, S_v]`.
    pub s_new: Tensor<R>,
}

/// Token `t` of a `[batch, seq, ..]` tensor, reshaped to `shape`.
pub(super) fn token_row<R: Runtime>(x: &Tensor<R>, t: usize, shape: &[usize]) -> Result<Tensor<R>> {
    Ok(x.narrow(1, t, 1)?.contiguous()?.reshape(shape)?)
}

/// Replay the forward over all `dims.seq` tokens.
///
/// Holds `2 * seq` states of `[batch, H, S_k, S_v]` at once.
pub(super) fn replay_forward<R, C>(
    client: &C,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    state: &Tensor<R>,
    dims: GdnDims,
) -> Result<Vec<Saved<R>>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + MatmulOps<R>,
{
    let GdnDims {
        batch,
        seq,
        heads,
        s_k,
        s_v,
    } = dims;
    let gate = [batch, heads, 1, 1];
    let mut saved = Vec::with_capacity(seq);
    let mut s = state.clone();
    for t in 0..seq {
        let k_t = token_row(k, t, &[batch, heads, 1, s_k])?;
        let v_t = token_row(v, t, &[batch, heads, 1, s_v])?;
        let beta_t = token_row(beta, t, &gate)?;
        let decay = client.exp(&token_row(g, t, &gate)?)?;

        let sbar = client.mul(&s, &decay)?;
        let p = client.matmul(&k_t, &sbar)?;
        let resid = client.sub(&v_t, &p)?;
        let d = client.mul(&resid, &beta_t)?;
        let s_new = client.add(&sbar, &client.matmul(&k_t.transpose(2, 3)?, &d)?)?;

        s = s_new.clone();
        saved.push(Saved {
            k: k_t,
            beta: beta_t,
            decay,
            sbar,
            resid,
            d,
            s_new,
        });
    }
    Ok(saved)
}

#[cfg(test)]
mod tests {
    use super::super::common::check_gdn_shapes;
    use super::super::step::gdn_step_impl;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Every replayed state equals the state `gdn_step_impl` carries.
    #[test]
    fn replay_states_match_step() {
        let (client, device) = cpu_setup();
        let (batch, seq, heads, s_k, s_v) = (2, 5, 2, 3, 4);
        let wave = |n: usize, f: f64| -> Vec<f64> {
            (0..n).map(|i| ((i as f64 + 1.0) * f).sin()).collect()
        };
        let rows = batch * seq * heads;
        let mk = |data: Vec<f64>, shape: &[usize]| {
            Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap()
        };
        let q = mk(wave(rows * s_k, 0.37), &[batch, seq, heads, s_k]);
        let k = mk(wave(rows * s_k, 0.53), &[batch, seq, heads, s_k]);
        let v = mk(wave(rows * s_v, 0.71), &[batch, seq, heads, s_v]);
        let g_data = wave(rows, 0.29).iter().map(|x| -0.1 - x.abs()).collect();
        let g = mk(g_data, &[batch, seq, heads]);
        let beta_data = wave(rows, 0.43).iter().map(|x| 0.5 + 0.4 * x).collect();
        let beta = mk(beta_data, &[batch, seq, heads]);
        let state = mk(
            wave(batch * heads * s_k * s_v, 0.61),
            &[batch, heads, s_k, s_v],
        );

        let dims = check_gdn_shapes(&q, &k, &v, &g, &beta, &state).unwrap();
        let saved = replay_forward(&client, &k, &v, &g, &beta, &state, dims).unwrap();
        assert_eq!(saved.len(), seq);

        let mut s = state.clone();
        for (t, sv) in saved.iter().enumerate() {
            let tok = |x: &Tensor<CpuRuntime>| x.narrow(1, t, 1).unwrap();
            let (_, s_next) = gdn_step_impl(
                &client,
                &tok(&q),
                &tok(&k),
                &tok(&v),
                &tok(&g),
                &tok(&beta),
                &s,
            )
            .unwrap();
            let want = s_next.to_vec::<f64>();
            let got = sv.s_new.to_vec::<f64>();
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-12, "t={t} idx={i}: got {a}, want {b}");
            }
            s = s_next;
        }
    }
}
