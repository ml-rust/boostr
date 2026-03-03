//! AVX2 integer Q8_0 × Q8_K dot product
//!
//! Q8_0 block (34 bytes, 32 elements):
//!   [0..2]  d (f16), [2..34] qs (i8×32)
//!
//! Q8_K block (292 bytes, 256 elements):
//!   [0..4]  d (f32), [4..260] qs (i8×256), [260..292] bsums (i16×16)
//!
//! 8 Q8_0 blocks (32 elem each) map to 1 Q8_K block (256 elem).
//! Uses i8→i16 widening + `_mm256_madd_epi16` for signed×signed correctness.
//! No min correction needed (Q8_0 has no min offset, unlike Q4_K).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_SIZE: usize = 32;

/// Fused Q8_0 × Q8_K dot product using AVX2.
///
/// # Safety
/// Requires AVX2. `act_q8k` must contain valid Q8_K blocks, `weight` must contain Q8_0 blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fused_dot_q8_0_q8k_avx2(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    unsafe {
        let num_super_blocks = k / 256; // number of Q8_K blocks
        let mut sumf = 0.0f32;

        for sb in 0..num_super_blocks {
            let q8k_block = &act_q8k[sb * Q8K_BLOCK_BYTES..];
            let d_a = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap_unchecked());
            let q8k_qs = q8k_block.as_ptr().add(4);

            // 8 Q8_0 sub-blocks per Q8_K super-block
            for sub in 0..8 {
                let q8_0_idx = sb * 8 + sub;
                let q8_0_block = &weight[q8_0_idx * Q8_0_BLOCK_BYTES..];
                let d_w = f16::from_le_bytes([q8_0_block[0], q8_0_block[1]]).to_f32();

                let w_ptr = q8_0_block.as_ptr().add(2);
                let a_ptr = q8k_qs.add(sub * Q8_0_BLOCK_SIZE);

                // Widen i8→i16 (handles signed×signed correctly)
                let w_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr as *const __m128i));
                let a_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(a_ptr as *const __m128i));
                let w_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(16) as *const __m128i));
                let a_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(a_ptr.add(16) as *const __m128i));

                // madd_epi16: signed i16 × signed i16 → i32 (sums adjacent pairs)
                let dot_lo = _mm256_madd_epi16(w_lo, a_lo);
                let dot_hi = _mm256_madd_epi16(w_hi, a_hi);
                let dot32 = _mm256_add_epi32(dot_lo, dot_hi);

                // Horizontal sum of dot32
                let hi128 = _mm256_extracti128_si256(dot32, 1);
                let lo128 = _mm256_castsi256_si128(dot32);
                let sum128 = _mm_add_epi32(lo128, hi128);
                let sum64 = _mm_shuffle_epi32(sum128, 0x4E);
                let sum128 = _mm_add_epi32(sum128, sum64);
                let sum32 = _mm_shuffle_epi32(sum128, 0xB1);
                let sum128 = _mm_add_epi32(sum128, sum32);
                let dot_result = _mm_cvtsi128_si32(sum128);

                sumf += d_w * d_a * dot_result as f32;
            }
        }

        sumf
    }
}

/// Dispatch wrapper
pub fn fused_dot_q8_0_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fused_dot_q8_0_q8k_avx2(act_q8k, weight, k) };
        }
    }

    fused_dot_q8_0_q8k_scalar(act_q8k, weight, k)
}

/// Scalar fallback
fn fused_dot_q8_0_q8k_scalar(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    let num_super_blocks = k / 256;
    let mut sumf = 0.0f32;

    for sb in 0..num_super_blocks {
        let q8k_block = &act_q8k[sb * Q8K_BLOCK_BYTES..];
        let d_a = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap());
        let q8k_qs = &q8k_block[4..260];

        for sub in 0..8 {
            let q8_0_idx = sb * 8 + sub;
            let q8_0_block = &weight[q8_0_idx * Q8_0_BLOCK_BYTES..];
            let d_w = f16::from_le_bytes([q8_0_block[0], q8_0_block[1]]).to_f32();
            let w_qs = &q8_0_block[2..34];

            let mut dot = 0i32;
            for l in 0..Q8_0_BLOCK_SIZE {
                dot += (w_qs[l] as i8 as i32) * (q8k_qs[sub * Q8_0_BLOCK_SIZE + l] as i8 as i32);
            }

            sumf += d_w * d_a * dot as f32;
        }
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::super::quantize_act_q8k::{Q8K_BLOCK_BYTES, quantize_f32_to_q8k};
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    #[test]
    fn test_fused_q8_0_q8k_vs_f32_dot() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let num_blocks = k / Q8_0_BLOCK_SIZE;
        let mut weight = vec![0u8; Q8_0_BLOCK_BYTES * num_blocks];
        for blk in 0..num_blocks {
            let base = blk * Q8_0_BLOCK_BYTES;
            weight[base..base + 2]
                .copy_from_slice(&f16::from_f32(0.5 + blk as f32 * 0.1).to_le_bytes());
            for i in 0..32 {
                weight[base + 2 + i] = ((blk * 17 + i * 7) % 256) as u8;
            }
        }

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q8_0_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q8_0(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.02 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    #[test]
    fn test_fused_q8_0_q8k_large() {
        let k = 4096;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.003).sin()).collect();

        let num_blocks = k / Q8_0_BLOCK_SIZE;
        let mut weight = vec![0u8; Q8_0_BLOCK_BYTES * num_blocks];
        for blk in 0..num_blocks {
            let base = blk * Q8_0_BLOCK_BYTES;
            weight[base..base + 2]
                .copy_from_slice(&f16::from_f32(0.3 + (blk as f32 * 0.01) % 1.0).to_le_bytes());
            for i in 0..32 {
                weight[base + 2 + i] = ((blk * 13 + i * 31) % 256) as u8;
            }
        }

        let num_q8k_blocks = k / 256;
        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * num_q8k_blocks];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q8_0_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q8_0(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.03 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }
}
