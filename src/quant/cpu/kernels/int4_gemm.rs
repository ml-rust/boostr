//! AWQ INT4 GEMM CPU kernel
//!
//! AWQ packing: 8 INT4 values per u32, non-sequential bit positions.
//! Shifts: [0, 16, 4, 20, 8, 24, 12, 28]
//!
//! Inner loop SIMD-accelerated on x86-64 AVX2: unpack 8 nibbles → dequant → FMA.
//! Parallelized across output columns via rayon for decode (M=1).

use rayon::prelude::*;

/// AWQ bit-shift table for unpacking 8 INT4 values from a u32.
/// AWQ packs with order [0,2,4,6,1,3,5,7], so to extract column j
/// the shift is: [0, 16, 4, 20, 8, 24, 12, 28][j]
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
#[allow(clippy::too_many_arguments)]
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
                int4_gemm_f32_avx2_par(
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

#[allow(clippy::too_many_arguments)]
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

/// AVX2 kernel parallelized across output column groups.
///
/// For decode (M=1): each rayon task computes a chunk of output columns,
/// iterating over all K for its columns. This is column-parallel
/// (vs the old K-major pattern which was sequential across N).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn int4_gemm_f32_avx2_par(
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

    // For M=1 (decode), parallelize across output columns
    // For M>1 (prefill), parallelize across rows
    if m == 1 {
        let inp_row = &input[..k];

        // Chunk size: 64 columns (8 packed u32s) per task — good cache locality
        let chunk_packed = 8usize; // 64 output columns per chunk
        let num_chunks = n_packed.div_ceil(chunk_packed);

        // Use par_chunks_mut on output to avoid Send issues
        let chunks: Vec<(usize, usize)> = (0..num_chunks)
            .map(|c| {
                let start_pack = c * chunk_packed;
                let end_pack = (start_pack + chunk_packed).min(n_packed);
                (start_pack, end_pack)
            })
            .collect();

        // Parallel over column chunks
        output[..n]
            .par_chunks_mut(chunk_packed * 8)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let (start_pack, end_pack) = chunks[chunk_idx];
                let chunk_n = (end_pack - start_pack) * 8;
                out_chunk[..chunk_n].fill(0.0);

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
                        for pack_j in start_pack..end_pack {
                            let packed = qweight[ki * n_packed + pack_j];
                            let base_col = pack_j * 8;
                            let w_vec = unpack_dequant_awq_avx2(
                                packed,
                                group_scales,
                                group_zeros,
                                base_col,
                            );
                            let local_col = (pack_j - start_pack) * 8;
                            let out_ptr = out_chunk.as_mut_ptr().add(local_col);
                            let out_vec = _mm256_loadu_ps(out_ptr);
                            let result = _mm256_fmadd_ps(a_vec, w_vec, out_vec);
                            _mm256_storeu_ps(out_ptr, result);
                        }
                    }
                }
            });
    } else {
        // M > 1: parallelize across rows
        let row_outputs: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|row| {
                let inp_row = &input[row * k..][..k];
                let mut out_row = vec![0.0f32; n];

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
                            let w_vec = unpack_dequant_awq_avx2(
                                packed,
                                group_scales,
                                group_zeros,
                                base_col,
                            );
                            let out_ptr = out_row.as_mut_ptr().add(base_col);
                            let out_vec = _mm256_loadu_ps(out_ptr);
                            let result = _mm256_fmadd_ps(a_vec, w_vec, out_vec);
                            _mm256_storeu_ps(out_ptr, result);
                        }
                    }
                }

                out_row
            })
            .collect();

        for (row, row_out) in row_outputs.into_iter().enumerate() {
            output[row * n..][..n].copy_from_slice(&row_out);
        }
    }
}
