//! Fused INT4 triple-GEMM QKV projection CPU kernel
//!
//! Computes (input@Wq, input@Wk, input@Wv) in a single pass over input.

use super::int4_gemm::AWQ_SHIFTS;

/// Extract the i-th INT4 value from AWQ-packed u32.
#[inline(always)]
fn unpack_int4_awq(packed: u32, idx: usize) -> u32 {
    (packed >> AWQ_SHIFTS[idx]) & 0xF
}

/// Fused INT4 QKV: computes Q, K, V projections in a single pass.
///
/// All weights in AWQ INT4 format.
/// input [M, K], Q weight [K, Nq/8], K/V weights [K, Nkv/8]
#[allow(clippy::too_many_arguments)]
pub fn fused_int4_qkv_f32(
    input: &[f32],
    qw_q: &[u32],
    sc_q: &[f32],
    zr_q: &[f32],
    qw_k: &[u32],
    sc_k: &[f32],
    zr_k: &[f32],
    qw_v: &[u32],
    sc_v: &[f32],
    zr_v: &[f32],
    out_q: &mut [f32],
    out_k: &mut [f32],
    out_v: &mut [f32],
    m: usize,
    k: usize,
    nq: usize,
    nkv: usize,
    group_size: usize,
) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(out_q.len(), m * nq);
    debug_assert_eq!(out_k.len(), m * nkv);
    debug_assert_eq!(out_v.len(), m * nkv);

    let nq_packed = nq / 8;
    let nkv_packed = nkv / 8;

    for row in 0..m {
        let inp_row = &input[row * k..][..k];
        let oq = &mut out_q[row * nq..][..nq];
        let ok = &mut out_k[row * nkv..][..nkv];
        let ov = &mut out_v[row * nkv..][..nkv];
        oq.fill(0.0);
        ok.fill(0.0);
        ov.fill(0.0);

        for ki in 0..k {
            let a = inp_row[ki];
            if a == 0.0 {
                continue;
            }
            let group = ki / group_size;

            // Q projection
            for pack_j in 0..nq_packed {
                let packed = qw_q[ki * nq_packed + pack_j];
                let base_col = pack_j * 8;
                for sub in 0..8 {
                    let col = base_col + sub;
                    let q = unpack_int4_awq(packed, sub) as f32;
                    let w = (q - zr_q[group * nq + col]) * sc_q[group * nq + col];
                    oq[col] += a * w;
                }
            }

            // K projection
            for pack_j in 0..nkv_packed {
                let packed = qw_k[ki * nkv_packed + pack_j];
                let base_col = pack_j * 8;
                for sub in 0..8 {
                    let col = base_col + sub;
                    let q = unpack_int4_awq(packed, sub) as f32;
                    let w = (q - zr_k[group * nkv + col]) * sc_k[group * nkv + col];
                    ok[col] += a * w;
                }
            }

            // V projection
            for pack_j in 0..nkv_packed {
                let packed = qw_v[ki * nkv_packed + pack_j];
                let base_col = pack_j * 8;
                for sub in 0..8 {
                    let col = base_col + sub;
                    let q = unpack_int4_awq(packed, sub) as f32;
                    let w = (q - zr_v[group * nkv + col]) * sc_v[group * nkv + col];
                    ov[col] += a * w;
                }
            }
        }
    }
}
