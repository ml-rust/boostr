//! NEON fused dequant+dot for Q4_K format
//!
//! Q4_K block layout (256 elements, 144 bytes):
//!   [0..2]   d (f16 scale)
//!   [2..4]   dmin (f16 minimum)
//!   [4..16]  sc (12-byte packed 6-bit scales+mins for 8 sub-blocks)
//!   [16..144] qs (128 bytes of 4-bit quantized values, 2 per byte)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use half::f16;

use super::super::super::dequant_k_quants::unpack_q4k_q5k_scales;
use super::dot_f32::hsum_f32_neon;

const F32_LANES: usize = 4;

/// Fused dequant+dot for Q4_K using NEON.
///
/// # Safety
/// Requires NEON. Caller must ensure act.len() >= k and blocks covers k/256 blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fused_dot_q4k_neon(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;
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
        let qs = &block[16..144];
        let act_block = &act[b * BLOCK_SIZE..];

        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;

            let chunk = j / 2;
            let is_high = j % 2 == 1;
            let qs_base = chunk * 32;

            let act_sub = &act_block[j * 32..];
            let dl_vec = vdupq_n_f32(dl);
            let ml_vec = vdupq_n_f32(ml);
            let mask_0f = vdupq_n_u32(0x0F);

            // Process 32 elements in 8 groups of 4
            for g in 0..8 {
                let l_base = g * 4;

                // Load 4 bytes of quantized data, zero-extend to u32
                let q0 = qs[qs_base + l_base] as u32;
                let q1 = qs[qs_base + l_base + 1] as u32;
                let q2 = qs[qs_base + l_base + 2] as u32;
                let q3 = qs[qs_base + l_base + 3] as u32;

                let raw = vcreate_u32(q0 as u64 | (q1 as u64) << 32);
                let raw_hi = vcreate_u32(q2 as u64 | (q3 as u64) << 32);
                let raw256 = vcombine_u32(raw, raw_hi);

                // Extract nibbles
                let nibbles = if is_high {
                    vandq_u32(vshrq_n_u32::<4>(raw256), mask_0f)
                } else {
                    vandq_u32(raw256, mask_0f)
                };

                // Convert to f32
                let q_f32 = vcvtq_f32_u32(nibbles);

                // Load 4 activation values
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
