//! Chunked GDN backward entry point.

use super::super::chunk_layout::{from_chunks, gate_from_chunks, pad_seq, to_chunks};
use super::super::chunk_terms::chunk_inputs;
use super::super::common::{check_gdn_shapes, expect_shape};
use super::adjoint::chunk_end_adjoints;
use super::local_grads::local_grads;
use super::states::chunk_start_states;
use crate::error::Result;
use crate::ops::traits::architecture::gated_delta_net::GdnGrads;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, CumulativeOps, LinalgOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Backward of [`gdn_chunk_prefill_impl`](super::super::chunk::gdn_chunk_prefill_impl).
///
/// `d_o` is `dL/do`: `[batch, seq, H, S_v]`. `d_state_final` is `dL/dS_T`:
/// `[batch, H, S_k, S_v]`. `None` means the final state is not in the loss.
/// `segment_ids` is the forward's I32 `[batch, seq]` document map. The
/// stages read the masked chunk terms, so no gradient crosses a document
/// start. `dstate` reaches only the first document of each row.
///
/// Three stages:
///
/// 1. Forward sweep: checkpoint the state at the start of every chunk.
/// 2. Reverse sweep: carry the state adjoint from the last chunk to the
///    first. It is the only sequential dependency.
/// 3. Local stage: every input gradient, batched over all chunks.
///
/// Gradients equal those of
/// [`gdn_recurrent_backward`](super::super::recurrent_bwd::gdn_recurrent_backward)
/// up to rounding. The forward terms run in the input dtype. Only the UT
/// transform promotes F16 and BF16 to F32.
///
/// Memory: the local stage holds about twenty `[N, n_chunks, cs, cs]` and
/// `[N, n_chunks, cs, S]` tensors at once, with `N = batch * H`. Both sweeps
/// keep `n_chunks` states of `[N, S_k, S_v]`.
#[allow(
    clippy::too_many_arguments,
    reason = "six forward inputs, segment ids, two output gradients and the chunk size"
)]
pub fn gdn_chunk_backward_impl<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    state: &Tensor<R>,
    segment_ids: Option<&Tensor<R>>,
    d_o: &Tensor<R>,
    d_state_final: Option<&Tensor<R>>,
    chunk_size: usize,
) -> Result<GdnGrads<R>>
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
        + UtilityOps<R>
        + LinalgOps<R>
        + LinearAlgebraAlgorithms<R>
        + TensorOps<R>
        + CompareOps<R>
        + TypeConversionOps<R>,
{
    let dims = check_gdn_shapes(q, k, v, g, beta, state)?;
    let (batch, seq, heads, s_k, s_v) = (dims.batch, dims.seq, dims.heads, dims.s_k, dims.s_v);
    expect_shape(d_o, "d_o", &[batch, seq, heads, s_v])?;
    let state_shape = [batch, heads, s_k, s_v];
    if let Some(ds) = d_state_final {
        expect_shape(ds, "d_state_final", &state_shape)?;
    }

    let x = chunk_inputs(client, q, k, v, g, beta, segment_ids, dims, chunk_size)?;
    let (n, n_chunks, padded) = (x.n, x.n_chunks, x.padded);
    // Padded tokens get a zero output gradient.
    let d_o = to_chunks(&pad_seq(client, d_o, seq, padded)?, n, n_chunks, chunk_size)?;

    let s0 = state.contiguous()?.reshape(&[n, s_k, s_v])?;
    let states = chunk_start_states(client, &x.terms, &s0, n_chunks)?;

    let ds_final = match d_state_final {
        Some(ds) => ds.contiguous()?.reshape(&[n, s_k, s_v])?,
        None => Tensor::<R>::zeros(&[n, s_k, s_v], state.dtype(), state.device())?,
    };
    let adj = chunk_end_adjoints(client, &x.terms, &d_o, &ds_final, n_chunks)?;

    let scale = 1.0 / (s_k as f64).sqrt();
    let local = local_grads(client, &x, &states, &adj.ds_end, &d_o, scale)?;

    Ok(GdnGrads {
        dq: from_chunks(&local.dq, dims, padded)?,
        dk: from_chunks(&local.dk, dims, padded)?,
        dv: from_chunks(&local.dv, dims, padded)?,
        dg: gate_from_chunks(&local.dg, dims, padded)?,
        dbeta: gate_from_chunks(&local.dbeta, dims, padded)?,
        dstate: adj.dstate.reshape(&state_shape)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::recurrent_bwd::gdn_recurrent_backward;
    use super::*;
    use crate::test_utils::{Lcg, cpu_setup};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const BATCH: usize = 2;
    const HEADS: usize = 2;
    /// `S_k != S_v`, so a swapped transpose fails on shape or value.
    const S_K: usize = 4;
    const S_V: usize = 5;
    /// F64 bound, relative to `max(1, largest oracle entry)`.
    const TOL_F64: f64 = 1e-9;
    /// F32 bound, relative to `max(1, largest oracle entry)`.
    const TOL_F32: f64 = 1e-4;
    const NAMES: [&str; 6] = ["dq", "dk", "dv", "dg", "dbeta", "dstate"];

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
        g: (0.05, 1.0),
        beta: (0.1, 0.9),
        key_spread: None,
    };

    const WEAK_DECAY: Regime = Regime {
        g: (0.001, 0.02),
        beta: (0.1, 0.9),
        key_spread: None,
    };

    /// Weak decay and nearly parallel keys: `I + A` is far from the identity.
    const OVERLAPPING_KEYS: Regime = Regime {
        g: (0.001, 0.02),
        beta: (0.7, 0.9),
        key_spread: Some(0.02),
    };

    fn normalize(row: &[f64]) -> Vec<f64> {
        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        row.iter().map(|x| x / norm).collect()
    }

    fn unit_rows(rng: &mut Lcg, rows: usize, dim: usize, spread: Option<f64>) -> Vec<f64> {
        let center = normalize(
            &(0..dim)
                .map(|_| rng.uniform_f64(-1.0, 1.0))
                .collect::<Vec<_>>(),
        );
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f64> = match spread {
                None => (0..dim).map(|_| rng.uniform_f64(-1.0, 1.0)).collect(),
                Some(eps) => center
                    .iter()
                    .map(|c| c + eps * rng.uniform_f64(-1.0, 1.0))
                    .collect(),
            };
            out.extend(normalize(&row));
        }
        out
    }

    /// Shapes of q, k, v, g, beta, state, d_o, d_state_final.
    fn shapes(seq: usize) -> [Vec<usize>; 8] {
        [
            vec![BATCH, seq, HEADS, S_K],
            vec![BATCH, seq, HEADS, S_K],
            vec![BATCH, seq, HEADS, S_V],
            vec![BATCH, seq, HEADS],
            vec![BATCH, seq, HEADS],
            vec![BATCH, HEADS, S_K, S_V],
            vec![BATCH, seq, HEADS, S_V],
            vec![BATCH, HEADS, S_K, S_V],
        ]
    }

    /// Host data in the order of [`shapes`].
    fn problem(seed: u64, regime: Regime, seq: usize) -> [Vec<f64>; 8] {
        let mut rng = Lcg(seed);
        let rows = BATCH * seq * HEADS;
        let state_len = BATCH * HEADS * S_K * S_V;
        let q = unit_rows(&mut rng, rows, S_K, None);
        let k = unit_rows(&mut rng, rows, S_K, regime.key_spread);
        let mut uni = |n: usize, lo: f64, hi: f64| -> Vec<f64> {
            (0..n).map(|_| rng.uniform_f64(lo, hi)).collect()
        };
        let v = uni(rows * S_V, -1.0, 1.0);
        let g = uni(rows, regime.g.0, regime.g.1)
            .iter()
            .map(|x| -x)
            .collect();
        let beta = uni(rows, regime.beta.0, regime.beta.1);
        let state = uni(state_len, -0.5, 0.5);
        let d_o = uni(rows * S_V, -1.0, 1.0);
        let d_state = uni(state_len, -1.0, 1.0);
        [q, k, v, g, beta, state, d_o, d_state]
    }

    fn tensors(
        device: &CpuDevice,
        data: &[Vec<f64>; 8],
        seq: usize,
        dtype: DType,
    ) -> Vec<Tensor<CpuRuntime>> {
        data.iter()
            .zip(shapes(seq))
            .map(|(d, shape)| match dtype {
                DType::F32 => {
                    let d: Vec<f32> = d.iter().map(|x| *x as f32).collect();
                    Tensor::<CpuRuntime>::from_slice(&d, &shape, device).unwrap()
                }
                _ => Tensor::<CpuRuntime>::from_slice(d, &shape, device).unwrap(),
            })
            .collect()
    }

    fn host(t: &Tensor<CpuRuntime>) -> Vec<f64> {
        match t.dtype() {
            DType::F32 => t.to_vec::<f32>().iter().map(|x| *x as f64).collect(),
            _ => t.to_vec::<f64>(),
        }
    }

    fn fields(g: GdnGrads<CpuRuntime>) -> [Tensor<CpuRuntime>; 6] {
        [g.dq, g.dk, g.dv, g.dg, g.dbeta, g.dstate]
    }

    fn chunked(
        t: &[Tensor<CpuRuntime>],
        with_state: bool,
        chunk_size: usize,
    ) -> GdnGrads<CpuRuntime> {
        let (client, _) = cpu_setup();
        gdn_chunk_backward_impl(
            &client,
            &t[0],
            &t[1],
            &t[2],
            &t[3],
            &t[4],
            &t[5],
            None,
            &t[6],
            with_state.then_some(&t[7]),
            chunk_size,
        )
        .unwrap()
    }

    /// Chunked backward in `dtype` against the F64 sequential oracle.
    ///
    /// `with_state` adds `<S_T, d_state_final>` to the loss.
    fn check(
        seq: usize,
        chunk_size: usize,
        seed: u64,
        regime: Regime,
        with_state: bool,
        dtype: DType,
    ) {
        let (client, device) = cpu_setup();
        let data = problem(seed, regime, seq);
        let w = tensors(&device, &data, seq, DType::F64);
        let want = gdn_recurrent_backward(
            &client,
            &w[0],
            &w[1],
            &w[2],
            &w[3],
            &w[4],
            &w[5],
            &w[6],
            with_state.then_some(&w[7]),
        )
        .unwrap();
        let got = chunked(&tensors(&device, &data, seq, dtype), with_state, chunk_size);
        let tol = if dtype == DType::F32 {
            TOL_F32
        } else {
            TOL_F64
        };
        for (i, (g, w)) in fields(got).iter().zip(fields(want).iter()).enumerate() {
            assert_eq!(g.shape(), w.shape(), "{} shape", NAMES[i]);
            assert_eq!(g.dtype(), dtype, "{} dtype", NAMES[i]);
            let (g, w) = (host(g), host(w));
            let scale = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            let diff = g
                .iter()
                .zip(&w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                diff <= tol * scale,
                "seq={seq} chunk={chunk_size} {}: diff {diff} (bound {})",
                NAMES[i],
                tol * scale
            );
        }
    }

    #[test]
    fn seq_1_chunk_64() {
        check(1, 64, 0xc0b_0001, STRONG_DECAY, true, DType::F64);
    }

    #[test]
    fn seq_7_chunk_3_strong_decay() {
        check(7, 3, 0xc0b_0002, STRONG_DECAY, true, DType::F64);
    }

    #[test]
    fn seq_7_chunk_3_weak_decay_without_final_state() {
        check(7, 3, 0xc0b_0003, WEAK_DECAY, false, DType::F64);
    }

    #[test]
    fn seq_7_chunk_3_overlapping_keys() {
        check(7, 3, 0xc0b_0004, OVERLAPPING_KEYS, true, DType::F64);
    }

    #[test]
    fn seq_64_chunk_64_strong_decay() {
        check(64, 64, 0xc0b_0005, STRONG_DECAY, true, DType::F64);
    }

    #[test]
    fn seq_64_chunk_32_weak_decay() {
        check(64, 32, 0xc0b_0006, WEAK_DECAY, true, DType::F64);
    }

    #[test]
    fn seq_130_chunk_64_strong_decay() {
        check(130, 64, 0xc0b_0007, STRONG_DECAY, true, DType::F64);
    }

    #[test]
    fn seq_130_chunk_64_weak_decay_without_final_state() {
        check(130, 64, 0xc0b_0008, WEAK_DECAY, false, DType::F64);
    }

    #[test]
    fn seq_130_chunk_64_overlapping_keys() {
        check(130, 64, 0xc0b_0009, OVERLAPPING_KEYS, true, DType::F64);
    }

    #[test]
    fn seq_130_chunk_32_strong_decay_without_final_state() {
        check(130, 32, 0xc0b_000a, STRONG_DECAY, false, DType::F64);
    }

    #[test]
    fn seq_130_chunk_32_weak_decay() {
        check(130, 32, 0xc0b_000b, WEAK_DECAY, true, DType::F64);
    }

    #[test]
    fn seq_130_chunk_32_overlapping_keys() {
        check(130, 32, 0xc0b_000c, OVERLAPPING_KEYS, true, DType::F64);
    }

    /// 44 chunks: the reverse sweep crosses many chunk boundaries.
    #[test]
    fn seq_130_chunk_3_weak_decay() {
        check(130, 3, 0xc0b_000d, WEAK_DECAY, true, DType::F64);
    }

    #[test]
    fn f32_seq_130_chunk_64_strong_decay() {
        check(130, 64, 0xc0b_000e, STRONG_DECAY, true, DType::F32);
    }

    /// Padding to a chunk multiple leaves every gradient unchanged and adds
    /// no token rows: chunk 5 has no padding, chunk 4 pads 3, chunk 64 pads 59.
    #[test]
    fn padded_tokens_are_dropped() {
        let (_, device) = cpu_setup();
        let seq = 5;
        let data = problem(0xc0b_000f, WEAK_DECAY, seq);
        let t = tensors(&device, &data, seq, DType::F64);
        let exact = fields(chunked(&t, true, 5));
        for chunk_size in [4, 64] {
            let padded = fields(chunked(&t, true, chunk_size));
            for (i, (p, e)) in padded.iter().zip(exact.iter()).enumerate() {
                assert_eq!(p.shape(), e.shape(), "{} shape", NAMES[i]);
                for (a, b) in host(p).iter().zip(host(e)) {
                    let bound = 1e-12 * b.abs().max(1.0);
                    assert!(
                        (a - b).abs() <= bound,
                        "chunk={chunk_size} {}: {a} vs {b}",
                        NAMES[i]
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_arguments() {
        let (client, device) = cpu_setup();
        let seq = 7;
        let t = tensors(&device, &problem(1, STRONG_DECAY, seq), seq, DType::F64);
        let zeros = |s: &[usize]| Tensor::<CpuRuntime>::zeros(s, DType::F64, &device).unwrap();
        let run = |d_o: &Tensor<CpuRuntime>, ds: Option<&Tensor<CpuRuntime>>, cs: usize| {
            gdn_chunk_backward_impl(
                &client, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], None, d_o, ds, cs,
            )
        };
        assert!(run(&t[6], Some(&t[7]), 3).is_ok());
        assert!(run(&t[6], None, 0).is_err());
        assert!(run(&zeros(&[BATCH, seq, HEADS, S_K]), None, 3).is_err());
        assert!(run(&t[6], Some(&zeros(&[BATCH, HEADS, S_V, S_K])), 3).is_err());
    }
}
