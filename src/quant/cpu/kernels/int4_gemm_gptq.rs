//! GPTQ INT4 GEMM CPU kernel
//!
//! GPTQ layout: sequential 4-bit packing in u32, g_idx permutation,
//! packed qzeros. Different dequant formula from AWQ.

/// Extract the i-th INT4 value (unsigned, 0..15) from a sequentially packed u32.
#[inline(always)]
fn unpack_int4_seq(packed: u32, idx: usize) -> u32 {
    (packed >> (idx * 4)) & 0xF
}

/// GPTQ INT4 GEMM: input [M, K] × dequantized weight [K, N] → output [M, N]
///
/// Weight stored as qweight [K/8, N] packed u32, with per-group scales/qzeros and g_idx.
/// Dequant formula: `w = q * scale + zero`
/// (where zero = -(qzero * scale), effectively `w = (q - qzero) * scale` but computed differently)
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

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];
        out_row.fill(0.0);

        for ki in 0..k {
            let a = inp_row[ki];
            if a == 0.0 {
                continue;
            }

            let group = g_idx[ki] as usize;
            let pack_row = ki / 8;
            let sub_idx = ki % 8;

            for col in 0..n {
                // Unpack weight
                let packed = qweight[pack_row * n + col];
                let q = unpack_int4_seq(packed, sub_idx) as f32;

                // Unpack qzero
                let zero_pack = qzeros[group * n_packed_zeros + col / 8];
                let qzero = unpack_int4_seq(zero_pack, col % 8) as f32;

                let scale = scales[group * n + col];
                // GPTQ formula: w = (q - qzero) * scale
                let w = (q - qzero) * scale;
                out_row[col] += a * w;
            }
        }
    }
}
