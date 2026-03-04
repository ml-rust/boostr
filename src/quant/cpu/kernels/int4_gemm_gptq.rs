//! GPTQ INT4 GEMM CPU kernel
//!
//! GPTQ layout: sequential 4-bit packing in u32, g_idx permutation,
//! packed qzeros. K-outer loop with per-group scale/qzero caching.

use rayon::prelude::*;

/// Extract the i-th INT4 value (unsigned, 0..15) from a sequentially packed u32.
#[inline(always)]
fn unpack_int4_seq(packed: u32, idx: usize) -> u32 {
    (packed >> (idx * 4)) & 0xF
}

/// GPTQ INT4 GEMM: input [M, K] × dequantized weight [K, N] → output [M, N]
#[allow(clippy::too_many_arguments)]
pub fn int4_gemm_gptq_f32(
    input: &[f32],
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[f32],
    g_idx: &[i32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(qweight.len(), (k / 8) * n);
    debug_assert_eq!(g_idx.len(), k);
    debug_assert_eq!(output.len(), m * n);

    let n_packed_zeros = n / 8;
    let output_ptr = output.as_mut_ptr() as usize;
    let num_threads = rayon::current_num_threads();

    let cols_per_thread = n.div_ceil(num_threads).max(64);
    let n_chunks = n.div_ceil(cols_per_thread);

    (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
        let col_start = chunk_idx * cols_per_thread;
        let col_end = (col_start + cols_per_thread).min(n);
        let ncols = col_end - col_start;

        // Thread-local accumulator
        let mut acc = vec![0.0f32; m * ncols];

        // Cached scale and dequantized zero for current group
        let mut cached_scale = vec![0.0f32; ncols];
        let mut cached_qzero = vec![0.0f32; ncols];
        let mut prev_group = usize::MAX;

        for ki in 0..k {
            let group = g_idx[ki] as usize;
            let pack_row = ki / 8;
            let sub_shift = (ki % 8) * 4;

            // Reload scale/qzero cache when group changes
            if group != prev_group {
                prev_group = group;
                let scales_base = group * n;
                let qzeros_base = group * n_packed_zeros;

                for c in 0..ncols {
                    let col = col_start + c;
                    cached_scale[c] = scales[scales_base + col];
                    let zero_pack = qzeros[qzeros_base + col / 8];
                    cached_qzero[c] = (unpack_int4_seq(zero_pack, col % 8) + 1) as f32;
                }
            }

            let qw_base = pack_row * n + col_start;

            for row in 0..m {
                let a = input[row * k + ki];
                if a == 0.0 {
                    continue;
                }
                let acc_row = &mut acc[row * ncols..][..ncols];

                for c in 0..ncols {
                    let packed = qweight[qw_base + c];
                    let q = ((packed >> sub_shift) & 0xF) as f32;
                    acc_row[c] += a * (q - cached_qzero[c]) * cached_scale[c];
                }
            }
        }

        // Write results
        unsafe {
            let out = output_ptr as *mut f32;
            for row in 0..m {
                for c in 0..ncols {
                    *out.add(row * n + col_start + c) = acc[row * ncols + c];
                }
            }
        }
    });
}
