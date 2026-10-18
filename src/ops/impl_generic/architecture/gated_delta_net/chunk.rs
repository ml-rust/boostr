//! Chunked GDN prefill.
//!
//! Port of `build_delta_net_chunking` (llama.cpp,
//! `src/models/delta-net-base.cpp`), rewritten for the `[batch, H, S_k, S_v]`
//! state orientation documented on the trait. Every tensor below is
//! row-major; llama.cpp's `ne[0]` is the last axis here.

use super::chunk_layout::{chunk_of, from_chunks};
use super::chunk_terms::{ChunkTerms, Chunked, advance_state, chunk_inputs};
use super::common::check_gdn_shapes;
use crate::error::Result;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CumulativeOps, LinalgOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Chunked gated delta rule over a whole sequence.
///
/// Per chunk (rows `j`, `i` index tokens in the chunk):
///
/// ```text
/// g_cs    = cumsum(g)                                  within the chunk
/// decay   = exp(g_cs[j] - g_cs[i])  for i <= j          1 above the diagonal
/// A       = (k_b @ k^T) * decay     strictly lower       k_b = k * beta
/// KQ      = (q @ k^T) * decay       lower incl. diagonal
/// T       = (I + A)^-1                                  UT transform, by
///                                                       forward substitution
/// U       = T @ v_b                                     v_b = v * beta
/// W       = T @ (k_b * exp(g_cs))
/// Q_g     = q * exp(g_cs)
/// K_g     = k * exp(g_last - g_cs)
/// V_new   = U - W @ S
/// o       = Q_g @ S + KQ @ V_new
/// S       = S * exp(g_last) + K_g^T @ V_new
/// ```
///
/// Returns `(o: [batch, seq, H, S_v], state: [batch, H, S_k, S_v])`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_prefill_impl<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    state: &Tensor<R>,
    chunk_size: usize,
) -> Result<(Tensor<R>, Tensor<R>)>
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
        + TensorOps<R>,
{
    let dims = check_gdn_shapes(q, k, v, g, beta, state)?;
    let (batch, heads, s_k, s_v) = (dims.batch, dims.heads, dims.s_k, dims.s_v);
    let Chunked {
        n,
        n_chunks,
        padded,
        terms,
        ..
    } = chunk_inputs(client, q, k, v, g, beta, dims, chunk_size)?;
    let ChunkTerms {
        u,
        w,
        kq,
        q_g,
        k_g_t,
        g_last_exp,
        ..
    } = terms;

    let mut s = state.contiguous()?.reshape(&[n, s_k, s_v])?;
    let mut outs: Vec<Tensor<R>> = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let w_c = chunk_of(&w, c)?;
        let u_c = chunk_of(&u, c)?;
        let kq_c = chunk_of(&kq, c)?;
        let q_g_c = chunk_of(&q_g, c)?;
        let k_g_t_c = chunk_of(&k_g_t, c)?;
        let gl = chunk_of(&g_last_exp, c)?.reshape(&[n, 1, 1])?;

        let o_inter = client.matmul(&q_g_c, &s)?;
        let (v_new, s_next) = advance_state(client, &s, &w_c, &u_c, &k_g_t_c, &gl)?;
        let o_intra = client.matmul(&kq_c, &v_new)?;
        outs.push(client.add(&o_inter, &o_intra)?);
        s = s_next;
    }

    let out_refs: Vec<&Tensor<R>> = outs.iter().collect();
    let o = client.cat(&out_refs, 1)?; // [N, padded, S_v]
    let o = from_chunks(&o, dims, padded)?;
    let s = s.reshape(&[batch, heads, s_k, s_v])?;
    Ok((o, s))
}

#[cfg(test)]
mod tests {
    use super::super::step::gdn_step_impl;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Deterministic 64-bit LCG; no rand dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }

        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }
    }

    /// How gates and keys are drawn.
    #[derive(Clone, Copy)]
    struct Regime {
        /// `g` is drawn from `-hi..-lo`.
        g: (f32, f32),
        /// `beta` is drawn from `lo..hi`.
        beta: (f32, f32),
        /// `None`: independent unit keys. `Some(eps)`: unit keys within
        /// about `eps` of one shared direction, so all keys nearly coincide.
        key_spread: Option<f32>,
    }

    /// Strong decay, independent keys.
    const STRONG_DECAY: Regime = Regime {
        g: (0.05, 1.0),
        beta: (0.1, 0.9),
        key_spread: None,
    };

    /// Weak decay, independent unit keys: the decay barely damps `A`.
    const WEAK_DECAY: Regime = Regime {
        g: (0.001, 0.02),
        beta: (0.1, 0.9),
        key_spread: None,
    };

    /// Weak decay and nearly parallel keys: every strictly lower entry of
    /// `A` is close to `beta`, so `I + A` is far from the identity.
    const OVERLAPPING_KEYS: Regime = Regime {
        g: (0.001, 0.02),
        beta: (0.7, 0.9),
        key_spread: Some(0.02),
    };

    struct Inputs {
        q: Tensor<CpuRuntime>,
        k: Tensor<CpuRuntime>,
        v: Tensor<CpuRuntime>,
        g: Tensor<CpuRuntime>,
        beta: Tensor<CpuRuntime>,
        state: Tensor<CpuRuntime>,
    }

    fn normalize(row: &[f32]) -> Vec<f32> {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        row.iter().map(|x| x / norm).collect()
    }

    fn unit_rows(rng: &mut Lcg, rows: usize, dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f32> = (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
            out.extend(normalize(&row));
        }
        out
    }

    /// Unit rows that each sit within about `eps` of one random direction.
    fn clustered_unit_rows(rng: &mut Lcg, rows: usize, dim: usize, eps: f32) -> Vec<f32> {
        let center: Vec<f32> = (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let center = normalize(&center);
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f32> = center
                .iter()
                .map(|c| c + eps * rng.uniform(-1.0, 1.0))
                .collect();
            out.extend(normalize(&row));
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn random_inputs(
        device: &CpuDevice,
        seed: u64,
        regime: Regime,
        batch: usize,
        seq: usize,
        heads: usize,
        s_k: usize,
        s_v: usize,
    ) -> Inputs {
        let mut rng = Lcg(seed);
        let rows = batch * seq * heads;
        let q = unit_rows(&mut rng, rows, s_k);
        let k = match regime.key_spread {
            None => unit_rows(&mut rng, rows, s_k),
            Some(eps) => clustered_unit_rows(&mut rng, rows, s_k, eps),
        };
        let v: Vec<f32> = (0..rows * s_v).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let g: Vec<f32> = (0..rows)
            .map(|_| -rng.uniform(regime.g.0, regime.g.1))
            .collect();
        let beta: Vec<f32> = (0..rows)
            .map(|_| rng.uniform(regime.beta.0, regime.beta.1))
            .collect();
        let state: Vec<f32> = (0..batch * heads * s_k * s_v)
            .map(|_| rng.uniform(-0.5, 0.5))
            .collect();
        let mk = |data: &[f32], shape: &[usize]| {
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
        };
        Inputs {
            q: mk(&q, &[batch, seq, heads, s_k]),
            k: mk(&k, &[batch, seq, heads, s_k]),
            v: mk(&v, &[batch, seq, heads, s_v]),
            g: mk(&g, &[batch, seq, heads]),
            beta: mk(&beta, &[batch, seq, heads]),
            state: mk(&state, &[batch, heads, s_k, s_v]),
        }
    }

    /// Run `gdn_step` token by token, carrying the state.
    fn sequential(client: &CpuClient, inp: &Inputs, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let mut state = inp.state.clone();
        let mut outs = Vec::with_capacity(seq);
        for t in 0..seq {
            let (o, s) = gdn_step_impl(
                client,
                &inp.q.narrow(1, t, 1).unwrap(),
                &inp.k.narrow(1, t, 1).unwrap(),
                &inp.v.narrow(1, t, 1).unwrap(),
                &inp.g.narrow(1, t, 1).unwrap(),
                &inp.beta.narrow(1, t, 1).unwrap(),
                &state,
            )
            .unwrap();
            outs.push(o);
            state = s;
        }
        let refs: Vec<&Tensor<CpuRuntime>> = outs.iter().collect();
        let o = client.cat(&refs, 1).unwrap();
        (o.to_vec::<f32>(), state.to_vec::<f32>())
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn max_abs(a: &[f32]) -> f32 {
        a.iter().map(|x| x.abs()).fold(0.0, f32::max)
    }

    /// Chunked prefill against the token-by-token recurrence.
    ///
    /// Both paths run in F32 with different operation orders. Each output
    /// and state entry is a sum of O(1) terms over at most `seq` steps, so
    /// the rounding gap stays near `seq` times the F32 unit roundoff
    /// (6e-8) relative to the largest entry. The bound 1e-4 times
    /// `max(1, largest entry)` leaves a margin of about 10x over that.
    /// A UT transform that loses precision on overlapping keys misses it by
    /// orders of magnitude.
    fn check_chunk_matches_step(seq: usize, chunk_size: usize, seed: u64, regime: Regime) {
        let (client, device) = cpu_setup();
        let (batch, heads, s_k, s_v) = (2, 3, 8, 8);
        let inp = random_inputs(&device, seed, regime, batch, seq, heads, s_k, s_v);

        let (o_chunk, s_chunk) = gdn_chunk_prefill_impl(
            &client, &inp.q, &inp.k, &inp.v, &inp.g, &inp.beta, &inp.state, chunk_size,
        )
        .unwrap();
        assert_eq!(o_chunk.shape(), &[batch, seq, heads, s_v]);
        assert_eq!(s_chunk.shape(), &[batch, heads, s_k, s_v]);

        let (o_step, s_step) = sequential(&client, &inp, seq);
        let o_tol = 1e-4 * max_abs(&o_step).max(1.0);
        let s_tol = 1e-4 * max_abs(&s_step).max(1.0);
        let o_diff = max_abs_diff(&o_chunk.to_vec::<f32>(), &o_step);
        let s_diff = max_abs_diff(&s_chunk.to_vec::<f32>(), &s_step);
        assert!(
            o_diff < o_tol,
            "seq={seq} chunk={chunk_size}: output diff {o_diff} (tol {o_tol})"
        );
        assert!(
            s_diff < s_tol,
            "seq={seq} chunk={chunk_size}: state diff {s_diff} (tol {s_tol})"
        );
    }

    #[test]
    fn chunk_matches_step_seq_130() {
        check_chunk_matches_step(130, 64, 0x5eed_0001, STRONG_DECAY);
    }

    #[test]
    fn chunk_matches_step_seq_64() {
        check_chunk_matches_step(64, 64, 0x5eed_0002, STRONG_DECAY);
    }

    #[test]
    fn chunk_matches_step_seq_1() {
        check_chunk_matches_step(1, 64, 0x5eed_0003, STRONG_DECAY);
    }

    /// A chunk size that is not a power of two.
    #[test]
    fn chunk_matches_step_odd_chunk() {
        check_chunk_matches_step(7, 3, 0x5eed_0004, STRONG_DECAY);
    }

    #[test]
    fn chunk_32_matches_step_seq_130() {
        check_chunk_matches_step(130, 32, 0x5eed_0005, STRONG_DECAY);
    }

    #[test]
    fn weak_decay_chunk_64_matches_step() {
        check_chunk_matches_step(130, 64, 0x5eed_0006, WEAK_DECAY);
    }

    #[test]
    fn weak_decay_chunk_32_matches_step() {
        check_chunk_matches_step(130, 32, 0x5eed_0007, WEAK_DECAY);
    }

    #[test]
    fn overlapping_keys_chunk_64_matches_step() {
        check_chunk_matches_step(130, 64, 0x5eed_0008, OVERLAPPING_KEYS);
    }

    #[test]
    fn overlapping_keys_chunk_32_matches_step() {
        check_chunk_matches_step(130, 32, 0x5eed_0009, OVERLAPPING_KEYS);
    }

    #[test]
    fn rejects_zero_chunk() {
        let (client, device) = cpu_setup();
        let inp = random_inputs(&device, 1, STRONG_DECAY, 1, 2, 1, 4, 4);
        let out = gdn_chunk_prefill_impl(
            &client, &inp.q, &inp.k, &inp.v, &inp.g, &inp.beta, &inp.state, 0,
        );
        assert!(out.is_err());
    }
}
