//! NF4 (Normal Float 4-bit) CPU kernels
//!
//! NF4 codebook: 16 values derived from normal distribution quantiles.
//! Used in QLoRA for 4-bit base model quantization.

/// NF4 codebook: 16 values from normal distribution quantiles.
/// Index 0 = 0.0, indices 1-7 = negative quantiles, indices 8-14 = positive, 15 = 1.0
pub const NF4_CODEBOOK: [f32; 16] = [
    0.0, -1.0, -0.6961928, -0.525073, -0.3949739, -0.2844144, -0.1848489, -0.0911179, 0.0796013,
    0.1609302, 0.2461123, 0.337912, 0.4407173, 0.562617, 0.7229568, 1.0,
];

/// Dequantize NF4 data to f32.
///
/// `nf4_data`: packed bytes, 2 indices per byte (low nibble first).
/// `absmax`: per-block scaling factors, shape [num_blocks].
/// `output`: f32 output, length = nf4_data.len() * 2.
pub fn nf4_dequant_f32(nf4_data: &[u8], absmax: &[f32], blocksize: usize, output: &mut [f32]) {
    let n = nf4_data.len() * 2;
    debug_assert_eq!(output.len(), n);
    debug_assert_eq!(absmax.len(), n.div_ceil(blocksize));

    for (i, &byte) in nf4_data.iter().enumerate() {
        let idx_lo = (byte & 0x0F) as usize;
        let idx_hi = ((byte >> 4) & 0x0F) as usize;

        let elem_lo = i * 2;
        let elem_hi = i * 2 + 1;

        let block_lo = elem_lo / blocksize;
        let block_hi = elem_hi / blocksize;

        output[elem_lo] = NF4_CODEBOOK[idx_lo] * absmax[block_lo];
        output[elem_hi] = NF4_CODEBOOK[idx_hi] * absmax[block_hi];
    }
}

/// Fused NF4 GEMM: input [M, K] × nf4_weight [N, K] → output [M, N]
///
/// Weight stored as nf4_data [N*K/2] u8, row-major [N, K] layout.
/// Dequantizes on-the-fly during dot product computation.
#[allow(clippy::too_many_arguments)]
pub fn nf4_gemm_f32(
    input: &[f32],
    nf4_weight: &[u8],
    absmax: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    blocksize: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(nf4_weight.len(), n * k / 2);
    debug_assert_eq!(output.len(), m * n);

    let k_packed = k / 2; // bytes per row of weight

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: is_x86_feature_detected! confirmed AVX2 + FMA are available at runtime,
            // satisfying the #[target_feature] contract of nf4_gemm_f32_avx2.
            unsafe {
                nf4_gemm_f32_avx2(
                    input, nf4_weight, absmax, output, m, k, n, blocksize, k_packed,
                );
            }
            return;
        }
    }

    nf4_gemm_f32_scalar(
        input, nf4_weight, absmax, output, m, k, n, blocksize, k_packed,
    );
}

#[allow(clippy::too_many_arguments)]
fn nf4_gemm_f32_scalar(
    input: &[f32],
    nf4_weight: &[u8],
    absmax: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    blocksize: usize,
    k_packed: usize,
) {
    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];

        for (col, out_elem) in out_row.iter_mut().enumerate() {
            let weight_row_start = col * k_packed;
            let weight_bytes = &nf4_weight[weight_row_start..][..k_packed];
            let absmax_row_start = col * (k / blocksize);

            let mut acc = 0.0f32;
            for (bi, &byte) in weight_bytes.iter().enumerate() {
                let idx_lo = (byte & 0x0F) as usize;
                let idx_hi = ((byte >> 4) & 0x0F) as usize;

                let elem_lo = bi * 2;
                let elem_hi = bi * 2 + 1;

                let block_lo = elem_lo / blocksize;
                let block_hi = elem_hi / blocksize;

                let w_lo = NF4_CODEBOOK[idx_lo] * absmax[absmax_row_start + block_lo];
                let w_hi = NF4_CODEBOOK[idx_hi] * absmax[absmax_row_start + block_hi];

                acc += inp_row[elem_lo] * w_lo + inp_row[elem_hi] * w_hi;
            }
            *out_elem = acc;
        }
    }
}

/// AVX2 NF4 GEMM: scalar codebook lookup, SIMD FMA accumulation.
/// Processes 8 bytes (16 elements) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn nf4_gemm_f32_avx2(
    input: &[f32],
    nf4_weight: &[u8],
    absmax: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    blocksize: usize,
    k_packed: usize,
) {
    use super::simd::dot_f32::hsum_f32_avx2;
    use std::arch::x86_64::*;

    // Process 8 bytes = 16 f32 elements per SIMD iteration
    let chunks = k_packed / 8;

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];

        for (col, out_val) in out_row.iter_mut().enumerate() {
            let weight_row_start = col * k_packed;
            let weight_bytes = &nf4_weight[weight_row_start..][..k_packed];
            let absmax_row_start = col * (k / blocksize);

            // _mm256_setzero_ps is a pure register op; AVX2 guaranteed by `#[target_feature]`.
            let mut acc = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();

            for chunk in 0..chunks {
                let base_bi = chunk * 8;
                let base_elem = base_bi * 2;

                // Scalar codebook lookups for 8 bytes → 16 dequantized weights
                let mut w_buf = [0.0f32; 16];
                for j in 0..8 {
                    let bi = base_bi + j;
                    let byte = weight_bytes[bi];
                    let elem_lo = bi * 2;
                    let elem_hi = bi * 2 + 1;
                    let block_lo = elem_lo / blocksize;
                    let block_hi = elem_hi / blocksize;
                    w_buf[j * 2] =
                        NF4_CODEBOOK[(byte & 0x0F) as usize] * absmax[absmax_row_start + block_lo];
                    w_buf[j * 2 + 1] = NF4_CODEBOOK[((byte >> 4) & 0x0F) as usize]
                        * absmax[absmax_row_start + block_hi];
                }

                // Load 16 activations and 16 weights, FMA accumulate.
                // _mm256_loadu_ps accepts unaligned pointers.
                // `inp_row[base_elem..]`: base_elem = chunk*16, chunks = k_packed/8, so
                // base_elem + 15 <= k_packed*2 - 1 = k - 1 = inp_row.len() - 1 (in bounds).
                // `w_buf` is a local [f32; 16] stack array; `w_buf[8..]` has exactly 8 elements.
                unsafe {
                    let a_lo = _mm256_loadu_ps(inp_row[base_elem..].as_ptr());
                    let w_lo = _mm256_loadu_ps(w_buf.as_ptr());
                    acc = _mm256_fmadd_ps(a_lo, w_lo, acc);

                    let a_hi = _mm256_loadu_ps(inp_row[base_elem + 8..].as_ptr());
                    let w_hi = _mm256_loadu_ps(w_buf[8..].as_ptr());
                    acc2 = _mm256_fmadd_ps(a_hi, w_hi, acc2);
                }
            }

            // `acc` and `acc2` are valid __m256 registers (zeroed or accumulated above).
            let mut result = unsafe { hsum_f32_avx2(_mm256_add_ps(acc, acc2)) };

            // Scalar tail
            for (bi, &byte) in weight_bytes[(chunks * 8)..].iter().enumerate() {
                let bi = chunks * 8 + bi;
                let elem_lo = bi * 2;
                let elem_hi = bi * 2 + 1;
                let block_lo = elem_lo / blocksize;
                let block_hi = elem_hi / blocksize;
                let w_lo =
                    NF4_CODEBOOK[(byte & 0x0F) as usize] * absmax[absmax_row_start + block_lo];
                let w_hi = NF4_CODEBOOK[((byte >> 4) & 0x0F) as usize]
                    * absmax[absmax_row_start + block_hi];
                result += inp_row[elem_lo] * w_lo + inp_row[elem_hi] * w_hi;
            }

            *out_val = result;
        }
    }
}
