//! AVX2 integer Q4_K × Q8_K dot product using `_mm256_maddubs_epi16`
//!
//! Replaces f32 FMA path with integer SIMD: 32 elements per instruction vs 8.
//! This is the single biggest CPU optimization for GGUF inference.
//!
//! Q4_K block (144 bytes, 256 elements):
//!   [0..2]    d (f16), [2..4] dmin (f16), [4..16] sc (12B packed scales/mins)
//!   [16..144] qs (128B, 4-bit nibbles, 2 per byte)
//!
//! Q8_K block (292 bytes, 256 elements):
//!   [0..4]     d (f32), [4..260] qs (i8×256), [260..292] bsums (i16×16)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::super::dequant_k_quants::unpack_q4k_q5k_scales;
use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

/// Fused Q4_K × Q8_K dot product using AVX2 maddubs.
///
/// # Safety
/// Requires AVX2. `act_q8k` must contain valid Q8_K blocks, `weight` must contain Q4_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_dot_q4k_q8k_avx2(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    unsafe {
        const Q4K_BLOCK_BYTES: usize = 144;
        const Q4K_BLOCK_SIZE: usize = 256;
        let num_blocks = k / Q4K_BLOCK_SIZE;

        let mut sumf = 0.0f32;

        for b in 0..num_blocks {
            let q4k = &weight[b * Q4K_BLOCK_BYTES..];
            let d = f16::from_le_bytes([q4k[0], q4k[1]]).to_f32();
            let dmin = f16::from_le_bytes([q4k[2], q4k[3]]).to_f32();
            let sc = &q4k[4..16];
            let qs = &q4k[16..144];

            let (scales, mins) = unpack_q4k_q5k_scales(sc);

            let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
            let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap_unchecked());
            let q8_qs = &q8k_block[4..260];
            let q8_bsums_ptr = q8k_block.as_ptr().add(260) as *const i16;

            let dall = d * d8;
            let dmin_all = dmin * d8;

            let mut sumi = 0i32;
            let mut sumi_mins = 0i32;

            for j in 0..4 {
                let q4_raw = _mm256_loadu_si256(qs.as_ptr().add(j * 32) as *const __m256i);
                let q4_lo = _mm256_and_si256(q4_raw, _mm256_set1_epi8(0x0F));
                let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_raw, 4), _mm256_set1_epi8(0x0F));

                let q8_lo = _mm256_loadu_si256(q8_qs.as_ptr().add(j * 64) as *const __m256i);
                let q8_hi = _mm256_loadu_si256(q8_qs.as_ptr().add(j * 64 + 32) as *const __m256i);

                let dot_lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
                let dot_hi = _mm256_maddubs_epi16(q4_hi, q8_hi);

                let scale_lo = _mm256_set1_epi16(scales[j * 2] as i16);
                let scale_hi = _mm256_set1_epi16(scales[j * 2 + 1] as i16);

                let p_lo = _mm256_madd_epi16(dot_lo, scale_lo);
                let p_hi = _mm256_madd_epi16(dot_hi, scale_hi);

                let p_sum = _mm256_add_epi32(p_lo, p_hi);

                let hi128 = _mm256_extracti128_si256(p_sum, 1);
                let lo128 = _mm256_castsi256_si128(p_sum);
                let sum128 = _mm_add_epi32(lo128, hi128);
                let sum64 = _mm_shuffle_epi32(sum128, 0x4E);
                let sum128 = _mm_add_epi32(sum128, sum64);
                let sum32 = _mm_shuffle_epi32(sum128, 0xB1);
                let sum128 = _mm_add_epi32(sum128, sum32);
                sumi += _mm_cvtsi128_si32(sum128);

                let bsum_lo =
                    (*q8_bsums_ptr.add(j * 4) as i32) + (*q8_bsums_ptr.add(j * 4 + 1) as i32);
                let bsum_hi =
                    (*q8_bsums_ptr.add(j * 4 + 2) as i32) + (*q8_bsums_ptr.add(j * 4 + 3) as i32);

                sumi_mins += (mins[j * 2] as i32) * bsum_lo + (mins[j * 2 + 1] as i32) * bsum_hi;
            }

            sumf += dall * sumi as f32 - dmin_all * sumi_mins as f32;
        }

        sumf
    }
}

/// Dispatch wrapper
pub fn fused_dot_q4k_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fused_dot_q4k_q8k_avx2(act_q8k, weight, k) };
        }
    }

    fused_dot_q4k_q8k_scalar(act_q8k, weight, k)
}

/// Scalar fallback
fn fused_dot_q4k_q8k_scalar(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    const Q4K_BLOCK_BYTES: usize = 144;
    const Q4K_BLOCK_SIZE: usize = 256;
    let num_blocks = k / Q4K_BLOCK_SIZE;

    let mut sumf = 0.0f32;

    for b in 0..num_blocks {
        let q4k = &weight[b * Q4K_BLOCK_BYTES..];
        let d = f16::from_le_bytes([q4k[0], q4k[1]]).to_f32();
        let dmin = f16::from_le_bytes([q4k[2], q4k[3]]).to_f32();
        let sc = &q4k[4..16];
        let qs = &q4k[16..144];

        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
        let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap());
        let q8_qs = &q8k_block[4..260];
        let q8_bsums = &q8k_block[260..292];

        let dall = d * d8;
        let dmin_all = dmin * d8;

        let mut sumi = 0i32;
        let mut sumi_mins = 0i32;

        for j in 0..8 {
            let chunk = j / 2;
            let is_high = j % 2 == 1;
            let qs_base = chunk * 32;

            let mut dot = 0i32;
            for l in 0..32 {
                let nibble = if is_high {
                    (qs[qs_base + l] >> 4) & 0x0F
                } else {
                    qs[qs_base + l] & 0x0F
                };
                dot += nibble as i32 * q8_qs[j * 32 + l] as i8 as i32;
            }
            sumi += dot * scales[j] as i32;

            // bsums: 2 sub-blocks of 16 elements for this 32-element group
            let bsum = i16::from_le_bytes(q8_bsums[j * 4..j * 4 + 2].try_into().unwrap()) as i32
                + i16::from_le_bytes(q8_bsums[j * 4 + 2..j * 4 + 4].try_into().unwrap()) as i32;
            sumi_mins += mins[j] as i32 * bsum;
        }

        sumf += dall * sumi as f32 - dmin_all * sumi_mins as f32;
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::super::quantize_act_q8k::{Q8K_BLOCK_BYTES, quantize_f32_to_q8k};
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    #[test]
    fn test_fused_q4k_q8k_vs_f32_dot() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        // Create Q4_K weight block
        let mut weight = [0u8; 144];
        weight[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        weight[2..4].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        weight[4..8].fill(0x03);
        weight[8..12].fill(0x02);
        weight[12..16].fill(0x10);
        weight[16..144].fill(0x73);

        // Quantize activation to Q8_K
        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q4k_q8k(&act_q8k, &weight, k);

        // Reference: dequant weight, dot with original f32 activation
        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q4k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        // Allow some quantization error from Q8_K
        assert!(
            (result - reference).abs() < reference.abs() * 0.02 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    #[test]
    fn test_fused_q4k_q8k_large() {
        let k = 4096;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let mut weight = vec![0u8; 144 * 16];
        for blk in 0..16 {
            let base = blk * 144;
            weight[base..base + 2]
                .copy_from_slice(&f16::from_f32(0.8 + blk as f32 * 0.05).to_le_bytes());
            weight[base + 2..base + 4]
                .copy_from_slice(&f16::from_f32(0.1 + blk as f32 * 0.01).to_le_bytes());
            weight[base + 4..base + 8].fill((blk as u8 % 10) + 1);
            weight[base + 8..base + 12].fill((blk as u8 % 5) + 1);
            for i in 16..144 {
                weight[base + i] = ((blk * 17 + i * 31) % 256) as u8;
            }
        }

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * 16];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q4k_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q4k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.02 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }
}
