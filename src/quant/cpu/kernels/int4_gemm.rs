//! AWQ INT4 GEMM CPU kernel
//!
//! AWQ packing: 8 INT4 values per u32, non-sequential bit positions.
//! Shifts: [0, 16, 4, 20, 8, 24, 12, 28]

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

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let out_row = &mut output[row * n..][..n];

        // Zero output
        out_row.fill(0.0);

        for ki in 0..k {
            let a = inp_row[ki];
            if a == 0.0 {
                continue;
            }
            let group = ki / group_size;

            // Process 8 output columns per packed u32
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
