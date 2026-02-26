//! NF4 (Normal Float 4-bit) CPU kernels
//!
//! NF4 codebook: 16 values derived from normal distribution quantiles.
//! Used in QLoRA for 4-bit base model quantization.

/// NF4 codebook: 16 values from normal distribution quantiles.
/// Index 0 = 0.0, indices 1-7 = negative quantiles, indices 8-14 = positive, 15 = 1.0
pub const NF4_CODEBOOK: [f32; 16] = [
    0.0, -1.0, -0.6961928, -0.5250730, -0.3949739, -0.2844144, -0.1848489, -0.0911179, 0.0796013,
    0.1609302, 0.2461123, 0.3379120, 0.4407173, 0.5626170, 0.7229568, 1.0,
];

/// Dequantize NF4 data to f32.
///
/// `nf4_data`: packed bytes, 2 indices per byte (low nibble first).
/// `absmax`: per-block scaling factors, shape [num_blocks].
/// `output`: f32 output, length = nf4_data.len() * 2.
pub fn nf4_dequant_f32(nf4_data: &[u8], absmax: &[f32], blocksize: usize, output: &mut [f32]) {
    let n = nf4_data.len() * 2;
    debug_assert_eq!(output.len(), n);
    debug_assert_eq!(absmax.len(), (n + blocksize - 1) / blocksize);

    for i in 0..nf4_data.len() {
        let byte = nf4_data[i];
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

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];

        for col in 0..n {
            let weight_row_start = col * k_packed;
            let weight_bytes = &nf4_weight[weight_row_start..][..k_packed];

            // absmax blocks are per-row: col * (k / blocksize)
            let absmax_row_start = col * (k / blocksize);

            let mut acc = 0.0f32;
            for bi in 0..k_packed {
                let byte = weight_bytes[bi];
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
            out_row[col] = acc;
        }
    }
}
