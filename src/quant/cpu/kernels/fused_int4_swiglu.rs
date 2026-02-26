//! Fused INT4 dual-GEMM + SwiGLU CPU kernel
//!
//! Computes `silu(input @ gate_w) * (input @ up_w)` in a single pass
//! over the input activation, reducing memory reads.

use super::int4_gemm::AWQ_SHIFTS;

/// Extract the i-th INT4 value from AWQ-packed u32.
#[inline(always)]
fn unpack_int4_awq(packed: u32, idx: usize) -> u32 {
    (packed >> AWQ_SHIFTS[idx]) & 0xF
}

/// SiLU activation: x * sigmoid(x)
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fused INT4 SwiGLU: silu(input @ gate_w) * (input @ up_w)
///
/// Both gate and up weights in AWQ INT4 format.
/// input [M, K], gate/up qweight [K, N/8], output [M, N]
pub fn fused_int4_swiglu_f32(
    input: &[f32],
    gate_qweight: &[u32],
    gate_scales: &[f32],
    gate_zeros: &[f32],
    up_qweight: &[u32],
    up_scales: &[f32],
    up_zeros: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(output.len(), m * n);

    let n_packed = n / 8;

    // Allocate temporary accumulators for gate and up
    let mut gate_acc = vec![0.0f32; n];
    let mut up_acc = vec![0.0f32; n];

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        gate_acc.fill(0.0);
        up_acc.fill(0.0);

        for ki in 0..k {
            let a = inp_row[ki];
            if a == 0.0 {
                continue;
            }
            let group = ki / group_size;

            for pack_j in 0..n_packed {
                let gate_packed = gate_qweight[ki * n_packed + pack_j];
                let up_packed = up_qweight[ki * n_packed + pack_j];
                let base_col = pack_j * 8;

                for sub in 0..8 {
                    let col = base_col + sub;
                    let gs = gate_scales[group * n + col];
                    let gz = gate_zeros[group * n + col];
                    let us = up_scales[group * n + col];
                    let uz = up_zeros[group * n + col];

                    let gq = unpack_int4_awq(gate_packed, sub) as f32;
                    let uq = unpack_int4_awq(up_packed, sub) as f32;

                    gate_acc[col] += a * (gq - gz) * gs;
                    up_acc[col] += a * (uq - uz) * us;
                }
            }
        }

        // Apply SwiGLU: silu(gate) * up
        let out_row = &mut output[row * n..][..n];
        for col in 0..n {
            out_row[col] = silu(gate_acc[col]) * up_acc[col];
        }
    }
}
