//! NEON fused dequant+dot for Q5_K format
//!
//! Q5_K block layout (256 elements, 176 bytes):
//!   [0..2]   d (f16 scale)
//!   [2..4]   dmin (f16 minimum)
//!   [4..16]  sc (12-byte packed 6-bit scales+mins for 8 sub-blocks)
//!   [16..48] qh (32 bytes, 1 high bit per element)
//!   [48..176] qs (128 bytes of 4-bit low nibbles, 2 per byte)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use half::f16;

use super::super::super::dequant_k_quants::unpack_q4k_q5k_scales;
use super::dot_f32::hsum_f32_neon;

/// Fused dequant+dot for Q5_K using NEON.
///
/// # Safety
/// Requires NEON. Caller must ensure act.len() >= k and blocks covers k/256 blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fused_dot_q5k_neon(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let num_blocks = k / BLOCK_SIZE;

    debug_assert!(act.len() >= k, "act.len() {} < k {}", act.len(), k);
    debug_assert!(
        blocks.len() >= num_blocks * BLOCK_BYTES,
        "blocks.len() {} < required {}",
        blocks.len(),
        num_blocks * BLOCK_BYTES
    );

    let mut total_acc = vdupq_n_f32(0.0);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let sc = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let act_block = &act[b * BLOCK_SIZE..];

        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;

            let act_sub = &act_block[j * 32..];
            let dl_vec = vdupq_n_f32(dl);
            let ml_vec = vdupq_n_f32(ml);

            // Process 32 elements in 8 groups of 4
            for g in 0..8 {
                let l_base = g * 4;

                // Build 5-bit values: low4 from qs + high1 from qh
                let mut q_vals = [0u32; 4];
                for l in 0..4 {
                    let idx = j * 32 + l_base + l;
                    let qs_idx = j * 16 + (l_base + l) / 2;
                    let low4 = if (l_base + l) % 2 == 0 {
                        qs[qs_idx] & 0x0F
                    } else {
                        (qs[qs_idx] >> 4) & 0x0F
                    };
                    let qh_byte = idx / 8;
                    let qh_bit = idx % 8;
                    let high1 = (qh[qh_byte] >> qh_bit) & 0x01;
                    q_vals[l] = (low4 | (high1 << 4)) as u32;
                }

                let q_u32 = vcombine_u32(
                    vcreate_u32(q_vals[0] as u64 | (q_vals[1] as u64) << 32),
                    vcreate_u32(q_vals[2] as u64 | (q_vals[3] as u64) << 32),
                );
                let q_f32 = vcvtq_f32_u32(q_u32);

                let a = vld1q_f32(act_sub.as_ptr().add(l_base));

                // Accumulate: dl * (a * q) - ml * a
                let aq = vmulq_f32(a, q_f32);
                total_acc = vfmaq_f32(total_acc, dl_vec, aq);
                total_acc = vsubq_f32(total_acc, vmulq_f32(ml_vec, a));
            }
        }
    }

    hsum_f32_neon(total_acc)
}
