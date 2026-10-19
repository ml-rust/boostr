//! Sequential GDN backward over the per-token recurrence.
//!
//! This is the reference gradient for faster backward paths. It replays the
//! forward of [`gdn_step_impl`](super::step::gdn_step_impl) one token at a
//! time and keeps every per-token state. Memory grows as
//! `seq * batch * H * S_k * S_v`.

use super::common::{check_gdn_shapes, expect_shape};
use super::recurrent_replay::{replay_forward, token_row};
use crate::error::Result;
use crate::ops::traits::architecture::gated_delta_net::GdnGrads;
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Put per-token rows back in token order and join them along `seq`.
///
/// `rows` is in reverse token order. Each row is reshaped to `shape`, which
/// has `1` on axis 1.
fn join_tokens<R, C>(client: &C, mut rows: Vec<Tensor<R>>, shape: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ShapeOps<R>,
{
    rows.reverse();
    let rows = rows
        .iter()
        .map(|r| -> Result<Tensor<R>> { Ok(r.contiguous()?.reshape(shape)?) })
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Tensor<R>> = rows.iter().collect();
    Ok(client.cat(&refs, 1)?)
}

/// Backward of the gated delta rule, one token at a time.
///
/// `d_o` is `dL/do`: `[batch, seq, H, S_v]`. `d_state_final` is `dL/dS_T`:
/// `[batch, H, S_k, S_v]`. `None` means the final state is not in the loss.
///
/// ```text
/// forward, per token (c = 1 / sqrt(S_k), vectors are columns):
///   a   = exp(g)            Sbar = a S_{t-1}
///   p   = Sbar^T k          d    = beta (v - p)
///   S_t = Sbar + k d^T      o    = (c q)^T S_t
///
/// backward, t = T .. 1, G = dS_T (zero when None):
///   G     += (c q) dO^T
///   dq     = c S_t dO
///   e      = G^T k
///   dv     = beta e
///   dbeta  = e . (v - p)
///   dk     = G d - beta Sbar e
///   dSbar  = G - k (beta e)^T
///   dg     = <dSbar, Sbar>_F
///   G      = a dSbar
/// dstate = G after t = 1
/// ```
///
/// The forward replay stores `Sbar` and `S_t` for every token.
#[allow(
    clippy::too_many_arguments,
    reason = "six forward inputs and two output gradients"
)]
pub fn gdn_recurrent_backward<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    state: &Tensor<R>,
    d_o: &Tensor<R>,
    d_state_final: Option<&Tensor<R>>,
) -> Result<GdnGrads<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ScalarOps<R>
        + MatmulOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + TensorOps<R>,
{
    let dims = check_gdn_shapes(q, k, v, g, beta, state)?;
    let (batch, seq, heads, s_k, s_v) = (dims.batch, dims.seq, dims.heads, dims.s_k, dims.s_v);
    expect_shape(d_o, "d_o", &[batch, seq, heads, s_v])?;
    let state_shape = [batch, heads, s_k, s_v];
    let mut grad_s = match d_state_final {
        Some(ds) => {
            expect_shape(ds, "d_state_final", &state_shape)?;
            ds.clone()
        }
        None => Tensor::<R>::zeros(&state_shape, state.dtype(), state.device())?,
    };

    let saved = replay_forward(client, k, v, g, beta, state, dims)?;
    let scale = 1.0 / (s_k as f64).sqrt();

    let mut dq = Vec::with_capacity(seq);
    let mut dk = Vec::with_capacity(seq);
    let mut dv = Vec::with_capacity(seq);
    let mut dg = Vec::with_capacity(seq);
    let mut dbeta = Vec::with_capacity(seq);
    for (t, sv) in saved.iter().enumerate().rev() {
        let cq = client.mul_scalar(&token_row(q, t, &[batch, heads, 1, s_k])?, scale)?;
        let do_t = token_row(d_o, t, &[batch, heads, 1, s_v])?;

        // G += (c q) dO^T : [B, H, S_k, 1] @ [B, H, 1, S_v]
        grad_s = client.add(&grad_s, &client.matmul(&cq.transpose(2, 3)?, &do_t)?)?;

        // dq = c S_t dO, as the row c dO^T S_t^T : [B, H, 1, S_k]
        let dq_t = client.matmul(&do_t, &sv.s_new.transpose(2, 3)?)?;
        dq.push(client.mul_scalar(&dq_t, scale)?);

        // e = G^T k, as the row k^T G : [B, H, 1, S_v]
        let e = client.matmul(&sv.k, &grad_s)?;
        let be = client.mul(&e, &sv.beta)?;
        dbeta.push(client.sum(&client.mul(&e, &sv.resid)?, &[3], true)?);

        // dk = G d - beta Sbar e, as rows d^T G^T - (beta e)^T Sbar^T
        let gd = client.matmul(&sv.d, &grad_s.transpose(2, 3)?)?;
        let sbar_e = client.matmul(&be, &sv.sbar.transpose(2, 3)?)?;
        dk.push(client.sub(&gd, &sbar_e)?);

        // dSbar = G - k (beta e)^T
        let dsbar = client.sub(&grad_s, &client.matmul(&sv.k.transpose(2, 3)?, &be)?)?;
        dg.push(client.sum(&client.mul(&dsbar, &sv.sbar)?, &[2, 3], true)?);
        dv.push(be);

        grad_s = client.mul(&dsbar, &sv.decay)?;
    }

    Ok(GdnGrads {
        dq: join_tokens(client, dq, &[batch, 1, heads, s_k])?,
        dk: join_tokens(client, dk, &[batch, 1, heads, s_k])?,
        dv: join_tokens(client, dv, &[batch, 1, heads, s_v])?,
        dg: join_tokens(client, dg, &[batch, 1, heads])?,
        dbeta: join_tokens(client, dbeta, &[batch, 1, heads])?,
        dstate: grad_s,
    })
}

#[cfg(test)]
mod tests {
    use super::super::step::gdn_step_impl;
    use super::*;
    use crate::test_utils::{Lcg, cpu_setup};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Central-difference step.
    const H: f64 = 1e-5;
    /// Allowed error relative to the largest analytic entry of each input.
    const REL_TOL: f64 = 1e-6;
    const NAMES: [&str; 6] = ["q", "k", "v", "g", "beta", "state"];

    /// How gates and keys are drawn.
    #[derive(Clone, Copy)]
    struct Regime {
        /// `g` is drawn from `-hi..-lo`.
        g: (f64, f64),
        /// `beta` is drawn from `lo..hi`.
        beta: (f64, f64),
        /// `None`: independent unit keys. `Some(eps)`: unit keys within
        /// about `eps` of one shared direction.
        key_spread: Option<f64>,
    }

    const STRONG_DECAY: Regime = Regime {
        g: (1.5, 2.5),
        beta: (0.1, 0.9),
        key_spread: None,
    };

    const WEAK_DECAY: Regime = Regime {
        g: (0.005, 0.015),
        beta: (0.1, 0.9),
        key_spread: None,
    };

    /// Weak decay and nearly parallel keys: each write overwrites the last.
    const OVERLAPPING_KEYS: Regime = Regime {
        g: (0.005, 0.015),
        beta: (0.7, 0.9),
        key_spread: Some(0.02),
    };

    const BETA_NEAR_ONE: Regime = Regime {
        g: (0.05, 1.0),
        beta: (0.95, 0.999),
        key_spread: None,
    };

    #[derive(Clone, Copy)]
    struct Dims {
        batch: usize,
        seq: usize,
        heads: usize,
        s_k: usize,
        s_v: usize,
    }

    /// `S_k != S_v`, so a swapped transpose fails on shape or value.
    const SEQ_7: Dims = Dims {
        batch: 2,
        seq: 7,
        heads: 2,
        s_k: 3,
        s_v: 4,
    };

    const SEQ_1: Dims = Dims { seq: 1, ..SEQ_7 };

    impl Dims {
        /// Shapes of q, k, v, g, beta, state.
        fn shapes(&self) -> [Vec<usize>; 6] {
            let Dims {
                batch,
                seq,
                heads,
                s_k,
                s_v,
            } = *self;
            [
                vec![batch, seq, heads, s_k],
                vec![batch, seq, heads, s_k],
                vec![batch, seq, heads, s_v],
                vec![batch, seq, heads],
                vec![batch, seq, heads],
                vec![batch, heads, s_k, s_v],
            ]
        }
    }

    /// Host inputs in the order of [`NAMES`], plus the loss weights.
    struct Problem {
        dims: Dims,
        inputs: [Vec<f64>; 6],
        w_o: Vec<f64>,
        w_s: Vec<f64>,
    }

    fn normalize(row: &[f64]) -> Vec<f64> {
        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        row.iter().map(|x| x / norm).collect()
    }

    fn unit_rows(rng: &mut Lcg, rows: usize, dim: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f64> = (0..dim).map(|_| rng.uniform_f64(-1.0, 1.0)).collect();
            out.extend(normalize(&row));
        }
        out
    }

    fn clustered_unit_rows(rng: &mut Lcg, rows: usize, dim: usize, eps: f64) -> Vec<f64> {
        let center = normalize(
            &(0..dim)
                .map(|_| rng.uniform_f64(-1.0, 1.0))
                .collect::<Vec<_>>(),
        );
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f64> = center
                .iter()
                .map(|c| c + eps * rng.uniform_f64(-1.0, 1.0))
                .collect();
            out.extend(normalize(&row));
        }
        out
    }

    fn problem(seed: u64, regime: Regime, dims: Dims) -> Problem {
        let mut rng = Lcg(seed);
        let rows = dims.batch * dims.seq * dims.heads;
        let state_len = dims.batch * dims.heads * dims.s_k * dims.s_v;
        let q = unit_rows(&mut rng, rows, dims.s_k);
        let k = match regime.key_spread {
            None => unit_rows(&mut rng, rows, dims.s_k),
            Some(eps) => clustered_unit_rows(&mut rng, rows, dims.s_k, eps),
        };
        let v = (0..rows * dims.s_v)
            .map(|_| rng.uniform_f64(-1.0, 1.0))
            .collect();
        let g = (0..rows)
            .map(|_| -rng.uniform_f64(regime.g.0, regime.g.1))
            .collect();
        let beta = (0..rows)
            .map(|_| rng.uniform_f64(regime.beta.0, regime.beta.1))
            .collect();
        let state = (0..state_len).map(|_| rng.uniform_f64(-0.5, 0.5)).collect();
        let w_o = (0..rows * dims.s_v)
            .map(|_| rng.uniform_f64(-1.0, 1.0))
            .collect();
        let w_s = (0..state_len).map(|_| rng.uniform_f64(-1.0, 1.0)).collect();
        Problem {
            dims,
            inputs: [q, k, v, g, beta, state],
            w_o,
            w_s,
        }
    }

    fn tensors(device: &CpuDevice, dims: Dims, inputs: &[Vec<f64>; 6]) -> Vec<Tensor<CpuRuntime>> {
        inputs
            .iter()
            .zip(dims.shapes())
            .map(|(data, shape)| Tensor::<CpuRuntime>::from_slice(data, &shape, device).unwrap())
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// `L = sum(o * w_o) + <S_T, w_s>`, the second term only `with_state`.
    fn loss(
        client: &CpuClient,
        device: &CpuDevice,
        p: &Problem,
        inputs: &[Vec<f64>; 6],
        with_state: bool,
    ) -> f64 {
        let t = tensors(device, p.dims, inputs);
        let mut state = t[5].clone();
        let mut outs = Vec::with_capacity(p.dims.seq);
        for step in 0..p.dims.seq {
            let tok = |x: &Tensor<CpuRuntime>| x.narrow(1, step, 1).unwrap();
            let (o, s) = gdn_step_impl(
                client,
                &tok(&t[0]),
                &tok(&t[1]),
                &tok(&t[2]),
                &tok(&t[3]),
                &tok(&t[4]),
                &state,
            )
            .unwrap();
            outs.push(o);
            state = s;
        }
        let refs: Vec<&Tensor<CpuRuntime>> = outs.iter().collect();
        let o = client.cat(&refs, 1).unwrap().to_vec::<f64>();
        let mut l = dot(&o, &p.w_o);
        if with_state {
            l += dot(&state.to_vec::<f64>(), &p.w_s);
        }
        l
    }

    /// Every analytic gradient entry against a central finite difference.
    fn check_grads(seed: u64, regime: Regime, dims: Dims, with_state: bool) {
        let (client, device) = cpu_setup();
        let p = problem(seed, regime, dims);
        let t = tensors(&device, dims, &p.inputs);
        let d_o = Tensor::<CpuRuntime>::from_slice(
            &p.w_o,
            &[dims.batch, dims.seq, dims.heads, dims.s_v],
            &device,
        )
        .unwrap();
        let w_s = Tensor::<CpuRuntime>::from_slice(&p.w_s, &dims.shapes()[5], &device).unwrap();
        let grads = gdn_recurrent_backward(
            &client,
            &t[0],
            &t[1],
            &t[2],
            &t[3],
            &t[4],
            &t[5],
            &d_o,
            with_state.then_some(&w_s),
        )
        .unwrap();
        let analytic = [
            grads.dq,
            grads.dk,
            grads.dv,
            grads.dg,
            grads.dbeta,
            grads.dstate,
        ];
        let shapes = dims.shapes();
        for (i, an) in analytic.iter().enumerate() {
            assert_eq!(an.shape(), shapes[i].as_slice(), "{} shape", NAMES[i]);
            let an = an.to_vec::<f64>();
            let scale = an.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for (j, &want) in an.iter().enumerate() {
                let mut plus = p.inputs.clone();
                plus[i][j] += H;
                let mut minus = p.inputs.clone();
                minus[i][j] -= H;
                let l_plus = loss(&client, &device, &p, &plus, with_state);
                let l_minus = loss(&client, &device, &p, &minus, with_state);
                let fd = (l_plus - l_minus) / (2.0 * H);
                let err = (want - fd).abs();
                assert!(
                    err <= REL_TOL * scale,
                    "{}[{j}]: analytic {want}, numeric {fd}, err {err} (scale {scale})",
                    NAMES[i]
                );
            }
        }
    }

    #[test]
    fn grads_strong_decay() {
        check_grads(0xb0d_0001, STRONG_DECAY, SEQ_7, true);
    }

    #[test]
    fn grads_weak_decay() {
        check_grads(0xb0d_0002, WEAK_DECAY, SEQ_7, true);
    }

    #[test]
    fn grads_overlapping_keys() {
        check_grads(0xb0d_0003, OVERLAPPING_KEYS, SEQ_7, true);
    }

    #[test]
    fn grads_beta_near_one() {
        check_grads(0xb0d_0004, BETA_NEAR_ONE, SEQ_7, true);
    }

    #[test]
    fn grads_single_token() {
        check_grads(0xb0d_0005, STRONG_DECAY, SEQ_1, true);
    }

    #[test]
    fn grads_single_token_weak_decay() {
        check_grads(0xb0d_0006, WEAK_DECAY, SEQ_1, true);
    }

    /// No final-state gradient: `G` starts at zero.
    #[test]
    fn grads_without_final_state() {
        check_grads(0xb0d_0007, WEAK_DECAY, SEQ_7, false);
    }

    /// A wrong `d_o` or `d_state_final` shape is an error.
    #[test]
    fn rejects_mismatched_gradients() {
        let (client, device) = cpu_setup();
        let p = problem(1, STRONG_DECAY, SEQ_7);
        let t = tensors(&device, SEQ_7, &p.inputs);
        let zeros = |s: &[usize]| Tensor::<CpuRuntime>::zeros(s, DType::F64, &device).unwrap();
        let run = |d_o: &Tensor<CpuRuntime>, ds: Option<&Tensor<CpuRuntime>>| {
            gdn_recurrent_backward(&client, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], d_o, ds)
        };
        let good_d_o = zeros(&[2, 7, 2, 4]);
        assert!(run(&good_d_o, None).is_ok());
        assert!(run(&zeros(&[2, 7, 2, 3]), None).is_err());
        assert!(run(&good_d_o, Some(&zeros(&[2, 2, 4, 3]))).is_err());
    }
}
