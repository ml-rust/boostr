//! Marlin-format INT4 GEMM CPU kernel
//!
//! Sequential 4-bit packing (not AWQ order), optimized for tensor core access.
//! Dequant formula: w = (q - 8) * scale + zero

/// Extract the i-th INT4 value (unsigned, 0..15) from a sequentially packed u32.
#[inline(always)]
fn unpack_int4_seq(packed: u32, idx: usize) -> u32 {
    (packed >> (idx * 4)) & 0xF
}

/// Marlin INT4 GEMM: input [M, K] × dequantized weight [K, N] → output [M, N]
///
/// Weight stored as [K/8, N] packed u32 (sequential 4-bit).
/// Dequant formula: `w = (q - 8) * scale + zero`
pub fn marlin_gemm_f32(
    input: &[f32],
    weight: &[u32],
    scales: &[f32],
    zeros: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(weight.len(), (k / 8) * n);
    debug_assert_eq!(output.len(), m * n);

    let k_packed = k / 8;
    let num_groups = k / group_size;
    debug_assert_eq!(scales.len(), num_groups * n);
    debug_assert_eq!(zeros.len(), num_groups * n);

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];
        out_row.fill(0.0);

        for pack_ki in 0..k_packed {
            for col in 0..n {
                let packed = weight[pack_ki * n + col];

                for sub in 0..8 {
                    let ki = pack_ki * 8 + sub;
                    let a = inp_row[ki];
                    if a == 0.0 {
                        continue;
                    }

                    let group = ki / group_size;
                    let q = unpack_int4_seq(packed, sub) as f32;
                    let scale = scales[group * n + col];
                    let zero = zeros[group * n + col];
                    let w = (q - 8.0) * scale + zero;
                    out_row[col] += a * w;
                }
            }
        }
    }
}
