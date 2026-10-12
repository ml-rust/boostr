//! Chunked GDN prefill.
//!
//! Port of `build_delta_net_chunking` (llama.cpp,
//! `src/models/delta-net-base.cpp`), rewritten for the `[batch, H, S_k, S_v]`
//! state orientation documented on the trait. Every tensor below is
//! row-major; llama.cpp's `ne[0]` is the last axis here.

use super::common::check_gdn_shapes;
use super::ut_transform::ut_transform;
use crate::error::{Error, Result};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CumulativeOps, LinalgOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Zero-pad `t` along `dim` 1 from `seq` to `padded` tokens.
fn pad_seq<R, C>(client: &C, t: &Tensor<R>, seq: usize, padded: usize) -> Result<Tensor<R>>
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
fn to_chunks<R: Runtime>(
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
fn gate_to_chunks<R: Runtime>(
    t: &Tensor<R>,
    n: usize,
    n_chunks: usize,
    chunk: usize,
) -> Result<Tensor<R>> {
    Ok(t.permute(&[0, 2, 1])?
        .contiguous()?
        .reshape(&[n, n_chunks, chunk])?)
}

/// Chunk `c` of a `[N, n_chunks, ..]` tensor as a contiguous `[N, ..]` tensor.
fn chunk_of<R: Runtime>(t: &Tensor<R>, c: usize) -> Result<Tensor<R>> {
    let mut shape = t.shape().to_vec();
    shape.remove(1);
    Ok(t.narrow(1, c, 1)?.contiguous()?.reshape(&shape)?)
}

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
    if chunk_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "chunk_size",
            reason: "must be >= 1".into(),
        });
    }
    let (batch, seq, heads, s_k, s_v) = (dims.batch, dims.seq, dims.heads, dims.s_k, dims.s_v);
    let n = batch * heads;
    let n_chunks = seq.div_ceil(chunk_size);
    let padded = n_chunks * chunk_size;
    let cs = chunk_size;
    let dtype = q.dtype();
    let device = q.device();

    // Pad to a chunk multiple. Padded tokens carry beta = 0 and g = 0, so
    // their rows of A, U and W vanish and the final state is untouched.
    let q = to_chunks(&pad_seq(client, q, seq, padded)?, n, n_chunks, cs)?;
    let q = client.mul_scalar(&q, 1.0 / (s_k as f64).sqrt())?;
    let k = to_chunks(&pad_seq(client, k, seq, padded)?, n, n_chunks, cs)?;
    let v = to_chunks(&pad_seq(client, v, seq, padded)?, n, n_chunks, cs)?;
    let g = gate_to_chunks(&pad_seq(client, g, seq, padded)?, n, n_chunks, cs)?;
    let beta = gate_to_chunks(&pad_seq(client, beta, seq, padded)?, n, n_chunks, cs)?
        .reshape(&[n, n_chunks, cs, 1])?;

    let k_b = client.mul(&k, &beta)?;
    let v_b = client.mul(&v, &beta)?;

    // Masks: [1, 1, cs, cs], broadcast over (N, n_chunks).
    let ones = Tensor::<R>::ones(&[cs, cs], dtype, device)?;
    let mask_lower = LinalgOps::tril(client, &ones, 0)?.reshape(&[1, 1, cs, cs])?;
    let mask_strict = LinalgOps::tril(client, &ones, -1)?.reshape(&[1, 1, cs, cs])?;

    // decay[j][i] = exp(g_cs[j] - g_cs[i]) for i <= j. The difference is
    // zeroed above the diagonal before exp (as ggml_tri does) so no
    // positive exponent is ever evaluated; A and KQ mask it out again.
    let g_cs = client.cumsum(&g, 2)?; // [N, n_chunks, cs]
    let g_j = g_cs.unsqueeze(3)?; // [N, n_chunks, cs, 1]
    let g_i = g_cs.unsqueeze(2)?; // [N, n_chunks, 1, cs]
    let diff = client.sub(&g_j, &g_i)?;
    let diff = client.mul(&diff, &mask_lower)?;
    let decay = client.exp(&diff)?;

    let k_t = k.transpose(2, 3)?;
    let a = client.matmul(&k_b, &k_t)?; // A[j][i] = beta_j k_j . k_i
    let a = client.mul(&client.mul(&a, &decay)?, &mask_strict)?;
    let kq = client.matmul(&q, &k_t)?; // KQ[j][i] = q_j . k_i
    let kq = client.mul(&client.mul(&kq, &decay)?, &mask_lower)?;

    let t = ut_transform(client, &a, n, n_chunks, cs)?; // [N, n_chunks, cs, cs]

    let u = client.matmul(&t, &v_b)?; // [N, n_chunks, cs, S_v]
    let g_exp = client.exp(&g_cs)?.unsqueeze(3)?; // [N, n_chunks, cs, 1]
    let w = client.matmul(&t, &client.mul(&k_b, &g_exp)?)?; // [N, n_chunks, cs, S_k]
    let q_g = client.mul(&q, &g_exp)?;

    let g_last = g_cs.narrow(2, cs - 1, 1)?.contiguous()?; // [N, n_chunks, 1]
    let g_last_exp = client.exp(&g_last)?;
    let g_diff_exp = client.exp(&client.sub(&g_last, &g_cs)?)?.unsqueeze(3)?;
    let k_g_t = client.mul(&k, &g_diff_exp)?.transpose(2, 3)?; // [N, n_chunks, S_k, cs]

    let mut s = state.contiguous()?.reshape(&[n, s_k, s_v])?;
    let mut outs: Vec<Tensor<R>> = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let w_c = chunk_of(&w, c)?;
        let u_c = chunk_of(&u, c)?;
        let kq_c = chunk_of(&kq, c)?;
        let q_g_c = chunk_of(&q_g, c)?;
        let k_g_t_c = chunk_of(&k_g_t, c)?;

        let v_prime = client.matmul(&w_c, &s)?; // [N, cs, S_v]
        let v_new = client.sub(&u_c, &v_prime)?;
        let o_intra = client.matmul(&kq_c, &v_new)?;
        let o_inter = client.matmul(&q_g_c, &s)?;
        outs.push(client.add(&o_inter, &o_intra)?);

        let ds = client.matmul(&k_g_t_c, &v_new)?; // [N, S_k, S_v]
        let gl = chunk_of(&g_last_exp, c)?.reshape(&[n, 1, 1])?;
        s = client.add(&client.mul(&s, &gl)?, &ds)?;
    }

    let out_refs: Vec<&Tensor<R>> = outs.iter().collect();
    let o = client.cat(&out_refs, 1)?; // [N, padded, S_v]
    let o = o
        .reshape(&[batch, heads, padded, s_v])?
        .narrow(2, 0, seq)?
        .permute(&[0, 2, 1, 3])?
        .contiguous()?;
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
