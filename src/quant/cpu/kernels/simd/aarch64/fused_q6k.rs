//! NEON fused dequant+dot for Q6_K format
//!
//! Q6_K block layout (256 elements, 210 bytes):
//!   [0..128]   ql (low 4 bits, packed 2 per byte for 256 elements)
//!   [128..192]  qh (high 2 bits, packed 4 per byte)
//!   [192..208]  sc (16 x i8 scales)
//!   [208..210]  d (f16 scale)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use half::f16;

use super::dot_f32::hsum_f32_neon;

/// Fused dequant+dot for Q6_K using NEON.
///
/// # Safety
/// Requires NEON. Caller must ensure act.len() >= k and blocks covers k/256 blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fused_dot_q6k_neon(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;
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
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc: &[i8] = std::slice::from_raw_parts(block[192..208].as_ptr() as *const i8, 16);
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let act_block = &act[b * BLOCK_SIZE..];

        // Process in two halves of 128 elements
        for n in 0..2 {
            let y_base = n * 128;
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;

            // Process 32 elements per sub-iteration, 4 outputs each
            for l in 0..32 {
                let is = l / 16;

                // Reconstruct 6-bit values
                let q1 = ((ql[ql_base + l] & 0x0F) | ((qh[qh_base + l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4))
                    as i8
                    - 32;
                let q3 =
                    ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4))
                    as i8
                    - 32;

                // Build NEON vector of 4 dequantized values
                let s1 = d * sc[sc_base + is] as f32;
                let s2 = d * sc[sc_base + is + 2] as f32;
                let s3 = d * sc[sc_base + is + 4] as f32;
                let s4 = d * sc[sc_base + is + 6] as f32;

                let dq = vcombine_f32(
                    vcreate_f32(
                        (s1 * q1 as f32).to_bits() as u64
                            | ((s2 * q2 as f32).to_bits() as u64) << 32,
                    ),
                    vcreate_f32(
                        (s3 * q3 as f32).to_bits() as u64
                            | ((s4 * q4 as f32).to_bits() as u64) << 32,
                    ),
                );

                // Load 4 activation values at positions [y_base+l, y_base+l+32, y_base+l+64, y_base+l+96]
                let a = vcombine_f32(
                    vcreate_f32(
                        (*act_block.as_ptr().add(y_base + l)).to_bits() as u64
                            | ((*act_block.as_ptr().add(y_base + l + 32)).to_bits() as u64) << 32,
                    ),
                    vcreate_f32(
                        (*act_block.as_ptr().add(y_base + l + 64)).to_bits() as u64
                            | ((*act_block.as_ptr().add(y_base + l + 96)).to_bits() as u64) << 32,
                    ),
                );

                total_acc = vfmaq_f32(total_acc, dq, a);
            }
        }
    }

    hsum_f32_neon(total_acc)
}
