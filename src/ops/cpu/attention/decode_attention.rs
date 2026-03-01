//! Fused CPU decode attention kernel for S_q = 1
//!
//! When generating one token at a time (decode), the standard attention path
//! (matmul → scale → softmax → matmul) creates multiple intermediate tensors
//! and dispatches through the generic op system. This kernel fuses everything
//! into a single pass per head:
//!
//!   1. Compute score[j] = dot(q[h], k[kv_h][j]) * scale  (AVX2 dot)
//!   2. Softmax: find max, compute exp, normalize
//!   3. output[h] = sum(weight[j] * v[kv_h][j])           (AVX2 FMA)
//!
//! No tensor allocations. No GQA expansion. Handles GQA by mapping query heads
//! to KV heads directly.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::error::{Error, Result};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Fused decode attention: Q [B, H, 1, D] × K [B, H_kv, S_k, D] → output [B, H, 1, D]
///
/// Returns (output, lse) where lse is a dummy [B, H, 1] tensor (needed for trait compat).
pub fn fused_decode_attention(
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let batch = q_shape[0];
    let seq_len_k = k_shape[2];

    debug_assert_eq!(q_shape[1], num_heads);
    debug_assert_eq!(q_shape[2], 1);
    debug_assert_eq!(q_shape[3], head_dim);
    debug_assert_eq!(k_shape[1], num_kv_heads);
    debug_assert_eq!(k_shape[3], head_dim);

    let scale = (head_dim as f32).sqrt().recip();
    let kv_group_size = num_heads / num_kv_heads;

    // Get raw data
    let q_data = unsafe { q.storage().as_host_slice::<f32>() };
    let k_data = unsafe { k.storage().as_host_slice::<f32>() };
    let v_data = unsafe { v.storage().as_host_slice::<f32>() };

    let mut output = vec![0.0f32; batch * num_heads * head_dim];
    let mut lse_data = vec![0.0f32; batch * num_heads];

    // Strides: [B, H, S, D] contiguous
    let q_stride_b = num_heads * head_dim; // S_q=1 so no S stride needed
    let k_stride_b = num_kv_heads * seq_len_k * head_dim;
    let k_stride_h = seq_len_k * head_dim;
    let v_stride_b = k_stride_b;
    let v_stride_h = k_stride_h;

    // Sequential over heads — attention is tiny compared to GEMVs for short sequences.
    // Avoids Rayon overhead and thread pool contention with GEMV parallelism.
    let mut scores = vec![0.0f32; seq_len_k];

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let q_offset = b * q_stride_b + h * head_dim;
            let k_base = b * k_stride_b + kv_h * k_stride_h;
            let v_base = b * v_stride_b + kv_h * v_stride_h;

            let q_row = &q_data[q_offset..q_offset + head_dim];

            // Phase 1: Compute QK scores with SIMD dot products
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..seq_len_k {
                let k_row = &k_data[k_base + j * head_dim..k_base + j * head_dim + head_dim];
                let score = dot_f32_simd(q_row, k_row) * scale;
                scores[j] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            // Phase 2: Softmax
            let mut sum_exp = 0.0f32;
            for j in 0..seq_len_k {
                let w = (scores[j] - max_score).exp();
                scores[j] = w;
                sum_exp += w;
            }

            // Phase 3: Accumulate weighted V into output
            let out_offset = b * num_heads * head_dim + h * head_dim;
            let out_row = &mut output[out_offset..out_offset + head_dim];
            out_row.fill(0.0);

            let inv_sum = 1.0 / sum_exp;
            for j in 0..seq_len_k {
                let w = scores[j] * inv_sum;
                let v_row = &v_data[v_base + j * head_dim..v_base + j * head_dim + head_dim];
                accumulate_weighted_simd(out_row, v_row, w);
            }

            lse_data[b * num_heads + h] = max_score + sum_exp.ln();
        }
    }

    let out_tensor =
        Tensor::<CpuRuntime>::from_slice(&output, &[batch, num_heads, 1, head_dim], q.device());
    let lse_tensor =
        Tensor::<CpuRuntime>::from_slice(&lse_data, &[batch, num_heads, 1], q.device());

    Ok((out_tensor, lse_tensor))
}

/// SIMD dot product of two f32 slices
#[inline]
fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2_fma(a.as_ptr(), b.as_ptr(), len) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// AVX2+FMA dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2_fma(a: *const f32, b: *const f32, len: usize) -> f32 {
    unsafe {
        const LANES: usize = 8;
        let chunks = len / LANES;
        let remainder = len % LANES;

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * LANES;
            let va = _mm256_loadu_ps(a.add(offset));
            let vb = _mm256_loadu_ps(b.add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut result = hsum_f32_avx2(acc);
        for i in 0..remainder {
            let offset = chunks * LANES + i;
            result += *a.add(offset) * *b.add(offset);
        }
        result
    }
}

/// Accumulate: out[i] += weight * v[i] using SIMD
#[inline]
fn accumulate_weighted_simd(out: &mut [f32], v: &[f32], weight: f32) {
    debug_assert_eq!(out.len(), v.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                accumulate_weighted_avx2(out.as_mut_ptr(), v.as_ptr(), weight, out.len());
            }
            return;
        }
    }

    for (o, &vi) in out.iter_mut().zip(v.iter()) {
        *o += weight * vi;
    }
}

/// AVX2+FMA weighted accumulation: out += weight * v
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accumulate_weighted_avx2(out: *mut f32, v: *const f32, weight: f32, len: usize) {
    unsafe {
        const LANES: usize = 8;
        let chunks = len / LANES;
        let remainder = len % LANES;
        let w_vec = _mm256_set1_ps(weight);

        for i in 0..chunks {
            let offset = i * LANES;
            let vo = _mm256_loadu_ps(out.add(offset));
            let vv = _mm256_loadu_ps(v.add(offset));
            let result = _mm256_fmadd_ps(w_vec, vv, vo);
            _mm256_storeu_ps(out.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * LANES + i;
            *out.add(offset) += weight * *v.add(offset);
        }
    }
}

/// Horizontal sum of AVX2 register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_f32_avx2(v: __m256) -> f32 {
    unsafe {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 0b_00_00_00_01);
        let sum32 = _mm_add_ss(sum64, hi32);
        _mm_cvtss_f32(sum32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn make_tensor(data: &[f32], shape: &[usize]) -> Tensor<CpuRuntime> {
        let device = CpuDevice::new();
        Tensor::<CpuRuntime>::from_slice(data, shape, &device)
    }

    #[test]
    fn test_decode_attention_basic() {
        // B=1, H=2, H_kv=2, S_k=3, D=4
        let q_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05).collect();
        let v_data: Vec<f32> = (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let q = make_tensor(&q_data, &[1, 2, 1, 4]);
        let k = make_tensor(&k_data, &[1, 2, 3, 4]);
        let v = make_tensor(&v_data, &[1, 2, 3, 4]);

        let (out, _lse) = fused_decode_attention(&q, &k, &v, 2, 2, 4).unwrap();
        assert_eq!(out.shape(), &[1, 2, 1, 4]);

        // Verify against reference: standard attention
        let out_data = out.to_vec::<f32>();
        // Output should be non-zero weighted combination of V
        let sum: f32 = out_data.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Output should be non-zero");
    }

    #[test]
    fn test_decode_attention_gqa() {
        // B=1, H=4, H_kv=2, S_k=3, D=4 (GQA ratio=2)
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05).collect();
        let v_data: Vec<f32> = (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let q = make_tensor(&q_data, &[1, 4, 1, 4]);
        let k = make_tensor(&k_data, &[1, 2, 3, 4]);
        let v = make_tensor(&v_data, &[1, 2, 3, 4]);

        let (out, _lse) = fused_decode_attention(&q, &k, &v, 4, 2, 4).unwrap();
        assert_eq!(out.shape(), &[1, 4, 1, 4]);

        // Heads 0,1 should share KV head 0; heads 2,3 should share KV head 1
        let out_data = out.to_vec::<f32>();
        // Heads sharing same KV head with same Q should produce same output
        // (different Q → different output, but same V weighting pattern)
        let sum: f32 = out_data.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_decode_attention_matches_standard() {
        use numr::ops::{ActivationOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps};
        use numr::runtime::cpu::CpuClient;

        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // B=1, H=2, H_kv=2, S_k=5, D=8
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let seq_len_k = 5;
        let scale = (head_dim as f64).sqrt().recip();

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.3).sin())
            .collect();
        let k_data: Vec<f32> = (0..num_kv_heads * seq_len_k * head_dim)
            .map(|i| ((i as f32) * 0.2).cos())
            .collect();
        let v_data: Vec<f32> = (0..num_kv_heads * seq_len_k * head_dim)
            .map(|i| ((i as f32) * 0.1 + 0.5).sin())
            .collect();

        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[1, num_heads, 1, head_dim], &device);
        let k = Tensor::<CpuRuntime>::from_slice(
            &k_data,
            &[1, num_kv_heads, seq_len_k, head_dim],
            &device,
        );
        let v = Tensor::<CpuRuntime>::from_slice(
            &v_data,
            &[1, num_kv_heads, seq_len_k, head_dim],
            &device,
        );

        // Fused kernel
        let (fused_out, _) =
            fused_decode_attention(&q, &k, &v, num_heads, num_kv_heads, head_dim).unwrap();

        // Reference: standard matmul path
        let k_t = k.transpose(-2isize, -1isize).unwrap().contiguous();
        let scores = client.matmul(&q, &k_t).unwrap();
        let scores = client.mul_scalar(&scores, scale).unwrap();
        let weights = client.softmax(&scores, -1).unwrap();
        let ref_out = client.matmul(&weights, &v).unwrap();

        let fused_data = fused_out.to_vec::<f32>();
        let ref_data = ref_out.to_vec::<f32>();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-4,
                "mismatch at {}: fused={}, ref={}, diff={}",
                i,
                f,
                r,
                (f - r).abs()
            );
        }
    }
}
