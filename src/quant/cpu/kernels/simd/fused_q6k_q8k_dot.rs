//! AVX2 integer Q6_K × Q8_K dot product using `_mm256_maddubs_epi16`
//!
//! Same maddubs pattern as Q4_K but with Q6_K unpacking (ql + qh → 6-bit values).
//!
//! Q6_K block (210 bytes, 256 elements):
//!   [0..128]   ql (low 4 bits)
//!   [128..192] qh (high 2 bits)
//!   [192..208] sc (16 × i8 scales)
//!   [208..210] d  (f16 overall scale)
//!
//! Q8_K block (292 bytes, 256 elements):
//!   [0..4] d (f32), [4..260] qs (i8×256), [260..292] bsums (i16×16)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

/// Fused Q6_K × Q8_K dot product using AVX2 maddubs.
///
/// # Safety
/// Requires AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_dot_q6k_q8k_avx2(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    unsafe {
        const Q6K_BLOCK_BYTES: usize = 210;
        const Q6K_BLOCK_SIZE: usize = 256;
        let num_blocks = k / Q6K_BLOCK_SIZE;

        let mut sumf = 0.0f32;

        for b in 0..num_blocks {
            let blk = &weight[b * Q6K_BLOCK_BYTES..];
            let ql = &blk[0..128];
            let qh = &blk[128..192];
            let sc = &blk[192..208];
            let d6 = f16::from_le_bytes([blk[208], blk[209]]).to_f32();

            let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
            let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap_unchecked());
            let q8_qs = &q8k_block[4..260];

            let dall = d6 * d8;
            let mut sumi = 0i32;

            // Q6_K has 2 halves of 128 elements.
            // Each half has 4 groups of 32 elements (q1-q4).
            // Each 32-element group has 2 scale sub-indices (is=0 for l<16, is=1 for l>=16).
            // We process 16 elements at a time using 128-bit SSE to match scale boundaries.

            for half in 0..2u32 {
                let ql_base = (half as usize) * 64;
                let qh_base = (half as usize) * 32;
                let sc_base = (half as usize) * 8;
                let q8_base = (half as usize) * 128;

                // Load 32 bytes of ql, ql[+32], and qh
                let ql_lo_vec = _mm256_loadu_si256(ql.as_ptr().add(ql_base) as *const __m256i);
                let ql_hi_vec = _mm256_loadu_si256(ql.as_ptr().add(ql_base + 32) as *const __m256i);
                let qh_vec = _mm256_loadu_si256(qh.as_ptr().add(qh_base) as *const __m256i);

                for g in 0..4 {
                    // Unpack 32 Q6_K unsigned values
                    let q6_vec = match g {
                        0 => {
                            let lo = _mm256_and_si256(ql_lo_vec, _mm256_set1_epi8(0x0F));
                            let hi = _mm256_slli_epi16(
                                _mm256_and_si256(qh_vec, _mm256_set1_epi8(0x03)),
                                4,
                            );
                            _mm256_or_si256(lo, hi)
                        }
                        1 => {
                            let lo = _mm256_and_si256(ql_hi_vec, _mm256_set1_epi8(0x0F));
                            let hi = _mm256_slli_epi16(
                                _mm256_and_si256(
                                    _mm256_srli_epi16(qh_vec, 2),
                                    _mm256_set1_epi8(0x03),
                                ),
                                4,
                            );
                            _mm256_or_si256(lo, hi)
                        }
                        2 => {
                            let lo = _mm256_and_si256(
                                _mm256_srli_epi16(ql_lo_vec, 4),
                                _mm256_set1_epi8(0x0F),
                            );
                            let hi = _mm256_and_si256(qh_vec, _mm256_set1_epi8(0x30));
                            _mm256_or_si256(lo, hi)
                        }
                        _ => {
                            let lo = _mm256_and_si256(
                                _mm256_srli_epi16(ql_hi_vec, 4),
                                _mm256_set1_epi8(0x0F),
                            );
                            let hi = _mm256_and_si256(
                                _mm256_srli_epi16(qh_vec, 2),
                                _mm256_set1_epi8(0x30),
                            );
                            _mm256_or_si256(lo, hi)
                        }
                    };

                    let q8_offset = q8_base + g * 32;
                    let q8_vec =
                        _mm256_loadu_si256(q8_qs.as_ptr().add(q8_offset) as *const __m256i);

                    let dot = _mm256_maddubs_epi16(q6_vec, q8_vec);
                    let ones = _mm256_set1_epi8(1);
                    let q8_sum = _mm256_maddubs_epi16(ones, q8_vec);
                    let corrected = _mm256_sub_epi16(dot, _mm256_slli_epi16(q8_sum, 5));

                    // Two scales per group: is=0 for first 16 elements, is=1 for last 16
                    let sc_offset = match g {
                        0 => sc_base,
                        1 => sc_base + 2,
                        2 => sc_base + 4,
                        _ => sc_base + 6,
                    };
                    let scale0 = *sc.as_ptr().add(sc_offset) as i8 as i16;
                    let scale1 = *sc.as_ptr().add(sc_offset + 1) as i8 as i16;

                    // Build scale vector: first 8 i16 lanes get scale0, last 8 get scale1
                    // __m256i lane layout: [0..7 from lo128, 8..15 from hi128]
                    // But maddubs produces i16 pairs from consecutive bytes, so:
                    // Lanes 0..7 correspond to bytes 0..15 (first 16 elements)
                    // Lanes 8..15 correspond to bytes 16..31 (last 16 elements)
                    let scale_lo = _mm_set1_epi16(scale0);
                    let scale_hi = _mm_set1_epi16(scale1);
                    let scale_vec = _mm256_set_m128i(scale_hi, scale_lo);

                    let p = _mm256_madd_epi16(corrected, scale_vec);

                    let hi128 = _mm256_extracti128_si256(p, 1);
                    let lo128 = _mm256_castsi256_si128(p);
                    let sum128 = _mm_add_epi32(lo128, hi128);
                    let sum64 = _mm_shuffle_epi32(sum128, 0x4E);
                    let sum128 = _mm_add_epi32(sum128, sum64);
                    let sum32 = _mm_shuffle_epi32(sum128, 0xB1);
                    let sum128 = _mm_add_epi32(sum128, sum32);
                    sumi += _mm_cvtsi128_si32(sum128);
                }
            }

            sumf += dall * sumi as f32;
        }

        sumf
    }
}

/// Dispatch wrapper
pub fn fused_dot_q6k_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fused_dot_q6k_q8k_avx2(act_q8k, weight, k) };
        }
    }

    fused_dot_q6k_q8k_scalar(act_q8k, weight, k)
}

/// Scalar fallback
fn fused_dot_q6k_q8k_scalar(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    const Q6K_BLOCK_BYTES: usize = 210;
    const Q6K_BLOCK_SIZE: usize = 256;
    let num_blocks = k / Q6K_BLOCK_SIZE;

    let mut sumf = 0.0f32;

    for b in 0..num_blocks {
        let blk = &weight[b * Q6K_BLOCK_BYTES..];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d6 = f16::from_le_bytes([blk[208], blk[209]]).to_f32();

        let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
        let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap());
        let q8_qs = &q8k_block[4..260];

        let dall = d6 * d8;
        let mut sumi = 0i32;

        for half in 0..2usize {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let sc_base = half * 8;
            let q8_base = half * 128;

            for l in 0..32 {
                let ql_lo = ql[ql_base + l];
                let ql_hi = ql[ql_base + 32 + l];
                let qh_val = qh[qh_base + l];
                let is = l / 16;

                let q1 = ((ql_lo & 0x0F) | ((qh_val & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql_hi & 0x0F) | (((qh_val >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql_lo >> 4) | (((qh_val >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql_hi >> 4) | (((qh_val >> 6) & 0x03) << 4)) as i32 - 32;

                let s1 = sc[sc_base + is] as i8 as i32;
                let s2 = sc[sc_base + is + 2] as i8 as i32;
                let s3 = sc[sc_base + is + 4] as i8 as i32;
                let s4 = sc[sc_base + is + 6] as i8 as i32;

                sumi += s1 * q1 * q8_qs[q8_base + l] as i8 as i32;
                sumi += s2 * q2 * q8_qs[q8_base + 32 + l] as i8 as i32;
                sumi += s3 * q3 * q8_qs[q8_base + 64 + l] as i8 as i32;
                sumi += s4 * q4 * q8_qs[q8_base + 96 + l] as i8 as i32;
            }
        }

        sumf += dall * sumi as f32;
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::super::quantize_act_q8k::{Q8K_BLOCK_BYTES, quantize_f32_to_q8k};
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    #[test]
    fn test_fused_q6k_q8k_vs_f32_dot() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let mut weight = [0u8; 210];
        weight[208..210].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        weight[192..208].fill(1);
        for i in 0..128 {
            weight[i] = ((i * 31) % 256) as u8;
        }
        for i in 0..64 {
            weight[128 + i] = ((i * 37) % 256) as u8;
        }

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q6k_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q6k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.02 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    #[test]
    fn test_fused_q6k_q8k_large() {
        let k = 4096;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let mut weight = vec![0u8; 210 * 16];
        for blk in 0..16 {
            let base = blk * 210;
            weight[base + 208..base + 210]
                .copy_from_slice(&f16::from_f32(0.8 + blk as f32 * 0.05).to_le_bytes());
            for i in 0..16 {
                weight[base + 192 + i] = ((i as i8 % 7) - 3) as u8;
            }
            for i in 0..128 {
                weight[base + i] = ((blk * 17 + i * 31) % 256) as u8;
            }
            for i in 0..64 {
                weight[base + 128 + i] = ((blk * 13 + i * 37) % 256) as u8;
            }
        }

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * 16];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q6k_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q6k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.02 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }
}
