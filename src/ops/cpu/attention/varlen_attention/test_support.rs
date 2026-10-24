//! Shared fixtures for the window tests of the CPU varlen kernels.

use super::mask::varlen_masked;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(super) const HEADS: usize = 4;
pub(super) const KV_HEADS: usize = 2;
pub(super) const HEAD_DIM: usize = 4;

/// Segment lengths `(seq_len_q, seq_len_k)` that cover equal, short-q and short-k cases.
pub(super) const SHAPES: [&[(usize, usize)]; 4] = [
    &[(4, 4)],
    &[(2, 5), (3, 3)],
    &[(5, 2), (1, 4)],
    &[(3, 3), (0, 0), (2, 6)],
];

pub(super) struct Packed {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub dout: Vec<f32>,
    pub cu_q: Vec<i32>,
    pub cu_k: Vec<i32>,
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

pub(super) fn packed(segments: &[(usize, usize)]) -> Packed {
    let cu_q = cumulative(segments.iter().map(|s| s.0));
    let cu_k = cumulative(segments.iter().map(|s| s.1));
    let tq = cu_q[cu_q.len() - 1] as usize;
    let tk = cu_k[cu_k.len() - 1] as usize;
    Packed {
        q: det(tq * HEADS * HEAD_DIM, 0.0),
        k: det(tk * KV_HEADS * HEAD_DIM, 1.3),
        v: det(tk * KV_HEADS * HEAD_DIM, 2.9),
        dout: det(tq * HEADS * HEAD_DIM, 4.1),
        cu_q,
        cu_k,
    }
}

pub(super) fn tensor(
    data: &[f32],
    tokens: usize,
    heads: usize,
    dev: &CpuDevice,
) -> Tensor<CpuRuntime> {
    match Tensor::<CpuRuntime>::from_slice(data, &[tokens, heads, HEAD_DIM], dev) {
        Ok(t) => t,
        Err(e) => panic!("test tensor: {e}"),
    }
}

/// Naive f64 forward with the shared predicate. Returns `(out, lse)`.
/// A row with no unmasked key gives zeros and `-inf`.
pub(super) fn naive_fwd(p: &Packed, causal: bool, window: usize) -> (Vec<f32>, Vec<f32>) {
    let tq = p.cu_q[p.cu_q.len() - 1] as usize;
    let mut out = vec![0.0f32; tq * HEADS * HEAD_DIM];
    let mut lse = vec![0.0f32; tq * HEADS];
    let scale = (HEAD_DIM as f64).sqrt().recip();
    for b in 0..p.cu_q.len() - 1 {
        let (q0, k0) = (p.cu_q[b] as usize, p.cu_k[b] as usize);
        let sq = p.cu_q[b + 1] as usize - q0;
        let sk = p.cu_k[b + 1] as usize - k0;
        let off = sk.saturating_sub(sq);
        for h in 0..HEADS {
            let kv = h / (HEADS / KV_HEADS);
            for i in 0..sq {
                let qb = ((q0 + i) * HEADS + h) * HEAD_DIM;
                let mut s: Vec<Option<f64>> = Vec::new();
                for j in 0..sk {
                    if varlen_masked(i, j, off, causal, window) {
                        s.push(None);
                        continue;
                    }
                    let kb = ((k0 + j) * KV_HEADS + kv) * HEAD_DIM;
                    let dot: f64 = (0..HEAD_DIM)
                        .map(|d| f64::from(p.q[qb + d]) * f64::from(p.k[kb + d]))
                        .sum();
                    s.push(Some(dot * scale));
                }
                let row = (q0 + i) * HEADS + h;
                let max = s.iter().flatten().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if max == f64::NEG_INFINITY {
                    lse[row] = f32::NEG_INFINITY;
                    continue;
                }
                let sum: f64 = s.iter().flatten().map(|x| (x - max).exp()).sum();
                for (j, x) in s.iter().enumerate() {
                    let Some(x) = x else { continue };
                    let w = (x - max).exp() / sum;
                    let vb = ((k0 + j) * KV_HEADS + kv) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        out[qb + d] += (w * f64::from(p.v[vb + d])) as f32;
                    }
                }
                lse[row] = (max + sum.ln()) as f32;
            }
        }
    }
    (out, lse)
}
