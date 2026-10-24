//! Sliding-window varlen attention: CPU against a window-aware reference, and WGPU against CPU.
//!
//! The rule under test, per segment with `key_offset = max(0, seq_len_k - seq_len_q)`:
//! key `j` is masked for query `i` when `(causal && j > key_offset + i)` or
//! `(window > 0 && j + window <= key_offset + i)`. A row with no unmasked key
//! outputs zeros with `lse = -inf` and adds no gradient.
//!
//! WGPU forward supports MHA only, so the WGPU forward cases use `kv_heads == heads`.
//! The WGPU backward cases also cover GQA, fed with the CPU output and lse.

use super::helpers::*;
use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

const HEAD_DIM: usize = 8;

struct Case {
    name: &'static str,
    heads: usize,
    kv_heads: usize,
    /// `(seq_len_q, seq_len_k)` per segment.
    segments: &'static [(usize, usize)],
}

const MHA_CASES: &[Case] = &[
    Case {
        name: "equal",
        heads: 2,
        kv_heads: 2,
        segments: &[(4, 4), (6, 6)],
    },
    Case {
        name: "short_q",
        heads: 2,
        kv_heads: 2,
        segments: &[(2, 5), (3, 4)],
    },
    Case {
        name: "short_k_empty_rows",
        heads: 2,
        kv_heads: 2,
        segments: &[(5, 2), (1, 3)],
    },
    Case {
        name: "empty_segment",
        heads: 2,
        kv_heads: 2,
        segments: &[(3, 3), (0, 0), (2, 6)],
    },
];

const GQA_CASES: &[Case] = &[
    Case {
        name: "gqa_equal",
        heads: 4,
        kv_heads: 2,
        segments: &[(4, 4), (6, 6)],
    },
    Case {
        name: "gqa_ragged",
        heads: 4,
        kv_heads: 2,
        segments: &[(2, 5), (5, 2), (3, 3)],
    },
    Case {
        name: "mqa_short_k",
        heads: 4,
        kv_heads: 1,
        segments: &[(5, 2), (2, 4)],
    },
];

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dout: Vec<f32>,
    cu_q: Vec<i32>,
    cu_k: Vec<i32>,
    total_q: usize,
    total_k: usize,
    max_seqlen: usize,
}

fn det(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 0.8)
        .collect()
}

fn cumulative(lens: impl Iterator<Item = usize>) -> Vec<i32> {
    let mut cu = vec![0i32];
    for l in lens {
        cu.push(cu[cu.len() - 1] + l as i32);
    }
    cu
}

impl Case {
    fn inputs(&self) -> Inputs {
        let cu_q = cumulative(self.segments.iter().map(|s| s.0));
        let cu_k = cumulative(self.segments.iter().map(|s| s.1));
        let total_q = cu_q[cu_q.len() - 1] as usize;
        let total_k = cu_k[cu_k.len() - 1] as usize;
        Inputs {
            q: det(total_q * self.heads * HEAD_DIM, 0.0),
            k: det(total_k * self.kv_heads * HEAD_DIM, 1.3),
            v: det(total_k * self.kv_heads * HEAD_DIM, 2.9),
            dout: det(total_q * self.heads * HEAD_DIM, 4.1),
            cu_q,
            cu_k,
            total_q,
            total_k,
            max_seqlen: self
                .segments
                .iter()
                .map(|s| s.0.max(s.1))
                .max()
                .unwrap_or(0),
        }
    }
}

fn masked(i: usize, j: usize, key_offset: usize, causal: bool, window: usize) -> bool {
    (causal && j > key_offset + i) || (window > 0 && j + window <= key_offset + i)
}

/// Naive packed reference with GQA, the window rule and the empty-row rule.
/// It repeats the CPU kernel's arithmetic order, so its output is bitwise comparable.
fn window_reference_fwd(c: &Case, x: &Inputs, causal: bool, window: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; x.total_q * c.heads * HEAD_DIM];
    let mut lse = vec![0.0f32; x.total_q * c.heads];
    let scale = (HEAD_DIM as f32).sqrt().recip();
    for b in 0..c.segments.len() {
        let (q0, k0) = (x.cu_q[b] as usize, x.cu_k[b] as usize);
        let (sq, sk) = c.segments[b];
        let key_offset = sk.saturating_sub(sq);
        for h in 0..c.heads {
            let kv_h = h / (c.heads / c.kv_heads);
            for qi in 0..sq {
                let qb = ((q0 + qi) * c.heads + h) * HEAD_DIM;
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(sk);
                for ki in 0..sk {
                    if masked(qi, ki, key_offset, causal, window) {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let kb = ((k0 + ki) * c.kv_heads + kv_h) * HEAD_DIM;
                    let mut dot = 0.0f32;
                    for d in 0..HEAD_DIM {
                        dot += x.q[qb + d] * x.k[kb + d];
                    }
                    let s = dot * scale;
                    max_score = max_score.max(s);
                    scores.push(s);
                }
                let row = (q0 + qi) * c.heads + h;
                if max_score == f32::NEG_INFINITY {
                    lse[row] = f32::NEG_INFINITY;
                    continue;
                }
                let mut sum_exp = 0.0f32;
                let mut exps = Vec::with_capacity(sk);
                for &s in &scores {
                    let e = (s - max_score).exp();
                    sum_exp += e;
                    exps.push(e);
                }
                for (ki, &e) in exps.iter().enumerate() {
                    let weight = e / sum_exp;
                    let vb = ((k0 + ki) * c.kv_heads + kv_h) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        out[qb + d] += weight * x.v[vb + d];
                    }
                }
                lse[row] = max_score + sum_exp.ln();
            }
        }
    }
    (out, lse)
}

struct Run {
    out: Vec<f32>,
    lse: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

fn run_cpu(c: &Case, x: &Inputs, causal: bool, window: usize) -> Run {
    let (client, dev) = setup_cpu();
    let t = |data: &[f32], tokens: usize, heads: usize| -> Tensor<CpuRuntime> {
        Tensor::from_slice(data, &[tokens, heads, HEAD_DIM], &dev).unwrap()
    };
    let (q, k, v) = (
        t(&x.q, x.total_q, c.heads),
        t(&x.k, x.total_k, c.kv_heads),
        t(&x.v, x.total_k, c.kv_heads),
    );
    let dout = t(&x.dout, x.total_q, c.heads);
    let batch = c.segments.len();
    let cq = det_i32_tensor(&x.cu_q, &[batch + 1], &dev);
    let ck = det_i32_tensor(&x.cu_k, &[batch + 1], &dev);
    let (out, lse) = client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cq,
            &ck,
            batch,
            c.heads,
            c.kv_heads,
            x.max_seqlen,
            x.max_seqlen,
            HEAD_DIM,
            causal,
            window,
        )
        .unwrap();
    let (dq, dk, dv) = client
        .varlen_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cq,
            &ck,
            &x.cu_q,
            &x.cu_k,
            batch,
            c.heads,
            c.kv_heads,
            x.max_seqlen,
            x.max_seqlen,
            HEAD_DIM,
            causal,
            window,
        )
        .unwrap();
    Run {
        out: out.to_vec::<f32>(),
        lse: lse.to_vec::<f32>(),
        dq: dq.to_vec::<f32>(),
        dk: dk.to_vec::<f32>(),
        dv: dv.to_vec::<f32>(),
    }
}

/// `-inf` must match `-inf`. Finite values use the standard tolerance.
#[cfg(feature = "wgpu")]
fn assert_lse_parity(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w == f32::NEG_INFINITY {
            assert_eq!(g, w, "{what}[{i}]: want -inf, got {g}");
        } else {
            assert_parity_f32_tol(&[g], &[w], &format!("{what}[{i}]"), 1e-5, 1e-6);
        }
    }
}

const WINDOWS: [usize; 4] = [1, 3, 4, 100];

#[test]
fn test_varlen_window_cpu_matches_reference_bitwise() {
    for c in MHA_CASES.iter().chain(GQA_CASES) {
        let x = c.inputs();
        for causal in [true, false] {
            for window in WINDOWS {
                let got = run_cpu(c, &x, causal, window);
                let (out, lse) = window_reference_fwd(c, &x, causal, window);
                let tag = format!("{} causal={causal} window={window}", c.name);
                assert_eq!(got.out, out, "{tag}: out");
                assert_eq!(got.lse, lse, "{tag}: lse");
                assert!(
                    got.out.iter().all(|v| v.is_finite()),
                    "{tag}: out not finite"
                );
                for g in got.dq.iter().chain(&got.dk).chain(&got.dv) {
                    assert!(g.is_finite(), "{tag}: gradient {g} not finite");
                }
            }
        }
    }
}

#[test]
fn test_varlen_window_at_least_max_seqlen_equals_no_window_bitwise() {
    for c in MHA_CASES.iter().chain(GQA_CASES) {
        let x = c.inputs();
        for causal in [true, false] {
            let base = run_cpu(c, &x, causal, 0);
            for window in [x.max_seqlen, x.max_seqlen + 5] {
                let got = run_cpu(c, &x, causal, window);
                let tag = format!("{} causal={causal} window={window}", c.name);
                assert_eq!(got.out, base.out, "{tag}: out");
                assert_eq!(got.lse, base.lse, "{tag}: lse");
                assert_eq!(got.dq, base.dq, "{tag}: dQ");
                assert_eq!(got.dk, base.dk, "{tag}: dK");
                assert_eq!(got.dv, base.dv, "{tag}: dV");
            }
        }
    }
}

#[test]
fn test_varlen_window_empty_rows_are_zero() {
    // 5 queries over 2 keys: with window 1 rows 2..5 keep no key, causal or not.
    let c = &MHA_CASES[2];
    let x = c.inputs();
    for causal in [true, false] {
        let got = run_cpu(c, &x, causal, 1);
        for row in 2..5 {
            for h in 0..c.heads {
                let r = row * c.heads + h;
                assert_eq!(got.lse[r], f32::NEG_INFINITY);
                let base = r * HEAD_DIM;
                assert!(got.out[base..base + HEAD_DIM].iter().all(|&v| v == 0.0));
                assert!(got.dq[base..base + HEAD_DIM].iter().all(|&v| v == 0.0));
            }
        }
    }
}

#[cfg(feature = "wgpu")]
fn run_wgpu(c: &Case, x: &Inputs, causal: bool, window: usize, cpu: &Run) {
    with_wgpu_backend(|client, device| {
        let t = |data: &[f32],
                 tokens: usize,
                 heads: usize|
         -> Tensor<numr::runtime::wgpu::WgpuRuntime> {
            Tensor::from_slice(data, &[tokens, heads, HEAD_DIM], &device).unwrap()
        };
        let tag = format!("{} causal={causal} window={window}", c.name);
        let batch = c.segments.len();
        let (q, k, v) = (
            t(&x.q, x.total_q, c.heads),
            t(&x.k, x.total_k, c.kv_heads),
            t(&x.v, x.total_k, c.kv_heads),
        );
        let dout = t(&x.dout, x.total_q, c.heads);
        let cq = Tensor::from_slice(&x.cu_q, &[batch + 1], &device).unwrap();
        let ck = Tensor::from_slice(&x.cu_k, &[batch + 1], &device).unwrap();

        // Forward runs on MHA cases only. Backward takes the CPU output and lse.
        let (out_w, lse_w) = if c.heads == c.kv_heads {
            let (out, lse) = client
                .varlen_attention_fwd(
                    &q,
                    &k,
                    &v,
                    &cq,
                    &ck,
                    batch,
                    c.heads,
                    c.kv_heads,
                    x.max_seqlen,
                    x.max_seqlen,
                    HEAD_DIM,
                    causal,
                    window,
                )
                .unwrap();
            assert_parity_f32(&out.to_vec::<f32>(), &cpu.out, &format!("{tag}: fwd out"));
            assert_lse_parity(&lse.to_vec::<f32>(), &cpu.lse, &format!("{tag}: fwd lse"));
            (out, lse)
        } else {
            (
                t(&cpu.out, x.total_q, c.heads),
                Tensor::from_slice(&cpu.lse, &[x.total_q, c.heads], &device).unwrap(),
            )
        };
        let (dq, dk, dv) = client
            .varlen_attention_bwd(
                &dout,
                &q,
                &k,
                &v,
                &out_w,
                &lse_w,
                &cq,
                &ck,
                &x.cu_q,
                &x.cu_k,
                batch,
                c.heads,
                c.kv_heads,
                x.max_seqlen,
                x.max_seqlen,
                HEAD_DIM,
                causal,
                window,
            )
            .unwrap();
        assert_parity_f32_relaxed(&dq.to_vec::<f32>(), &cpu.dq, &format!("{tag}: dQ"));
        assert_parity_f32_relaxed(&dk.to_vec::<f32>(), &cpu.dk, &format!("{tag}: dK"));
        assert_parity_f32_relaxed(&dv.to_vec::<f32>(), &cpu.dv, &format!("{tag}: dV"));
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_varlen_window_wgpu_matches_cpu() {
    for c in MHA_CASES.iter().chain(GQA_CASES) {
        let x = c.inputs();
        for causal in [true, false] {
            for window in WINDOWS {
                let cpu = run_cpu(c, &x, causal, window);
                run_wgpu(c, &x, causal, window, &cpu);
            }
        }
    }
}
