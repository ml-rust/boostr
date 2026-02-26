//! AWQ INT4 GEMM CPU kernel
//!
//! AWQ packing: 8 INT4 values per u32, non-sequential bit positions.
//! Shifts: [0, 16, 4, 20, 8, 24, 12, 28]
//!
//! Inner loop SIMD-accelerated on x86-64 AVX2: unpack 8 nibbles → dequant → FMA.

/// AWQ bit-shift table for unpacking 8 INT4 values from a u32.
pub const AWQ_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];

/// Extract the i-th INT4 value (unsigned, 0..15) from an AWQ-packed u32.
#[inline(always)]
fn unpack_int4_awq(packed: u32, idx: usize) -> u32 {
    (packed >> AWQ_SHIFTS[idx]) & 0xF
}

/// AWQ INT4 GEMM: input [M, K] × dequantized weight [K, N] → output [M, N]
///
/// Weight stored as qweight [K, N/8] packed u32, with per-group scales/zeros.
/// Dequant formula: `w = (q - zero) * scale`
pub fn int4_gemm_f32(
    input: &[f32],
    qweight: &[u32],
    scales: &[f32],
    zeros: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(qweight.len(), k * (n / 8));
    debug_assert_eq!(output.len(), m * n);

    let n_packed = n / 8;
    let num_groups = k / group_size;
    debug_assert_eq!(scales.len(), num_groups * n);
    debug_assert_eq!(zeros.len(), num_groups * n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                int4_gemm_f32_avx2(
                    input, qweight, scales, zeros, output, m, k, n, group_size, n_packed,
                );
            }
            return;
        }
    }

    int4_gemm_f32_scalar(
        input, qweight, scales, zeros, output, m, k, n, group_size, n_packed,
    );
}

fn int4_gemm_f32_scalar(
    input: &[f32],
    qweight: &[u32],
    scales: &[f32],
    zeros: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    n_packed: usize,
) {
    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];
        out_row.fill(0.0);

        for ki in 0..k {
            let a = inp_row[ki];
            if a == 0.0 {
                continue;
            }
            let group = ki / group_size;

            for pack_j in 0..n_packed {
                let packed = qweight[ki * n_packed + pack_j];
                let base_col = pack_j * 8;

                for sub in 0..8 {
                    let col = base_col + sub;
                    let q = unpack_int4_awq(packed, sub) as f32;
                    let scale = scales[group * n + col];
                    let zero = zeros[group * n + col];
                    let w = (q - zero) * scale;
                    out_row[col] += a * w;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn int4_gemm_f32_avx2(
    input: &[f32],
    qweight: &[u32],
    scales: &[f32],
    zeros: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    n_packed: usize,
) {
    use super::simd::int4_unpack::unpack_dequant_awq_avx2;
    use std::arch::x86_64::*;

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];
        out_row.fill(0.0);

        for ki in 0..k {
            let a = inp_row[ki];
            if a == 0.0 {
                continue;
            }
            let group = ki / group_size;
            let group_scales = scales[group * n..].as_ptr();
            let group_zeros = zeros[group * n..].as_ptr();

            unsafe {
                let a_vec = _mm256_set1_ps(a);
                for pack_j in 0..n_packed {
                    let packed = qweight[ki * n_packed + pack_j];
                    let base_col = pack_j * 8;
                    let w_vec =
                        unpack_dequant_awq_avx2(packed, group_scales, group_zeros, base_col);
                    let out_ptr = out_row.as_mut_ptr().add(base_col);
                    let out_vec = _mm256_loadu_ps(out_ptr);
                    let result = _mm256_fmadd_ps(a_vec, w_vec, out_vec);
                    _mm256_storeu_ps(out_ptr, result);
                }
            }
        }
    }
}
