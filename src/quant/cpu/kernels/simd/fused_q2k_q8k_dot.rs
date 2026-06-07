//! AVX2 integer Q2_K × Q8_K dot product
//!
//! Matches llama.cpp's `ggml_vec_dot_q2_K_q8_K` algorithm:
//! integer arithmetic per sub-block, only float at final accumulation.
//!
//! Q2_K block (84 bytes, 256 elements):
//!   [0..16]  scales (16 bytes, low 4 bits = sub-scale, high 4 bits = sub-min)
//!   [16..80] qs (64 bytes, 2-bit values packed 4 per byte)
//!   [80..82] d (f16 scale)
//!   [82..84] dmin (f16 minimum)
//!
//! Q8_K block (292 bytes, 256 elements):
//!   [0..4]     d (f32), [4..260] qs (i8×256), [260..292] bsums (i16×16)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

/// Fused Q2_K × Q8_K dot product using AVX2 maddubs.
///
/// # Safety
/// Requires AVX2. `act_q8k` must contain valid Q8_K blocks, `weight` must contain Q2_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_dot_q2k_q8k_avx2(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    unsafe {
        const Q2K_BLOCK_BYTES: usize = 84;
        const Q2K_BLOCK_SIZE: usize = 256;
        let num_blocks = k / Q2K_BLOCK_SIZE;

        let mut sumf = 0.0f32;

        // Constant masks
        let mask2 = _mm256_set1_epi8(0x03);

        for b in 0..num_blocks {
            let q2k = &weight[b * Q2K_BLOCK_BYTES..];
            let sc = &q2k[0..16];
            let qs = &q2k[16..80];
            let d = f16::from_le_bytes([q2k[80], q2k[81]]).to_f32();
            let dmin = f16::from_le_bytes([q2k[82], q2k[83]]).to_f32();

            let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
            let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap_unchecked());
            let q8_qs = &q8k_block[4..260];
            let bsums_ptr = q8k_block.as_ptr().add(260) as *const i16;

            let dall = d * d8;
            let dmin_all = dmin * d8;

            // Sum mins: bsums[j] * (sc[j] >> 4)  for j in 0..16
            let mut summs = 0i32;
            for (j, &scj) in sc.iter().enumerate() {
                let bsum = *bsums_ptr.add(j) as i32;
                summs += bsum * (scj >> 4) as i32;
            }

            // Integer dot product over 16 sub-blocks (2 outer × 4 shifts × 2 halves)
            // We iterate over 2 outer groups (n=0,1) and 4 shift levels (0,2,4,6).
            // Each iteration covers a 32-element group (sub-block A: 16 elements from
            // qs[n*32..][0..16], sub-block B: 16 elements from qs[n*32..][16..32]).
            // sub-block A uses sc[is] low 4 bits, sub-block B uses sc[is+1] low 4 bits.
            //
            // AVX2 approach: for each shift, load the full 32-byte qs slice for this outer
            // group, extract the 2-bit nibbles at the given shift as u8 (0-3), load the
            // corresponding 32 bytes of Q8_K values, and use maddubs to compute the dot.
            // Scale sub-block A (lo 128 bits) and sub-block B (hi 128 bits) with separate
            // i16 scale values applied via madd.

            let mut isum = 0i32;
            let mut is: usize = 0;
            let mut q8_offset: usize = 0;

            for n in 0..2usize {
                // 32 bytes of 2-bit packed weights for this outer group
                let qs_ptr = qs.as_ptr().add(n * 32);

                // Load all 32 bytes once; shift and mask per shift level
                let raw = _mm256_loadu_si256(qs_ptr as *const __m256i);

                // 4 shift levels: 0, 2, 4, 6
                // _mm256_srli_epi16 requires a compile-time constant immediate, so we
                // unroll via match rather than computing (shift * 2) at runtime.
                for shift in 0..4usize {
                    // Extract 2-bit nibbles at this shift → u8 values in [0, 3]
                    let shifted = match shift {
                        0 => raw,
                        1 => _mm256_srli_epi16(raw, 2),
                        2 => _mm256_srli_epi16(raw, 4),
                        _ => _mm256_srli_epi16(raw, 6),
                    };
                    let q2_vals = _mm256_and_si256(shifted, mask2);

                    // Load 32 bytes of Q8_K activations (i8 values in the first 16
                    // bytes correspond to sub-block A, next 16 to sub-block B)
                    let q8_vec =
                        _mm256_loadu_si256(q8_qs.as_ptr().add(q8_offset) as *const __m256i);

                    // maddubs: (u8 q2_vals) × (i8 q8_vec) → i16, accumulated pairwise
                    let dot16 = _mm256_maddubs_epi16(q2_vals, q8_vec);

                    // We now need to apply two different scales to the lo and hi 128-bit
                    // halves (sub-block A: lanes 0-7 of dot16, sub-block B: lanes 8-15).
                    let scale_a = (sc[is] & 0x0F) as i16;
                    let scale_b = (sc[is + 1] & 0x0F) as i16;
                    is += 2;

                    // Build scale vector: lo128 = scale_a repeated, hi128 = scale_b repeated
                    let scale_lo = _mm_set1_epi16(scale_a);
                    let scale_hi = _mm_set1_epi16(scale_b);
                    let scale_vec = _mm256_set_m128i(scale_hi, scale_lo);

                    // madd: i16 × i16 → i32 pairwise, with scale per lane
                    let p = _mm256_madd_epi16(dot16, scale_vec);

                    // Horizontal sum of the 8 i32 lanes
                    let hi128 = _mm256_extracti128_si256(p, 1);
                    let lo128 = _mm256_castsi256_si128(p);
                    let sum128 = _mm_add_epi32(lo128, hi128);
                    let sum64 = _mm_shuffle_epi32(sum128, 0x4E);
                    let sum128 = _mm_add_epi32(sum128, sum64);
                    let sum32 = _mm_shuffle_epi32(sum128, 0xB1);
                    let sum128 = _mm_add_epi32(sum128, sum32);
                    isum += _mm_cvtsi128_si32(sum128);

                    q8_offset += 32;
                }
            }
            sumf += dall * isum as f32 - dmin_all * summs as f32;
        }

        sumf
    }
}

/// Dispatch wrapper
pub fn fused_dot_q2k_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fused_dot_q2k_q8k_avx2(act_q8k, weight, k) };
        }
    }

    fused_dot_q2k_q8k_scalar(act_q8k, weight, k)
}

/// Scalar fallback — matches llama.cpp's `ggml_vec_dot_q2_K_q8_K` exactly.
fn fused_dot_q2k_q8k_scalar(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    const Q2K_BLOCK_BYTES: usize = 84;
    const Q2K_BLOCK_SIZE: usize = 256;
    let num_blocks = k / Q2K_BLOCK_SIZE;

    let mut sumf = 0.0f32;

    for b in 0..num_blocks {
        let q2k = &weight[b * Q2K_BLOCK_BYTES..];
        let sc = &q2k[0..16];
        let qs = &q2k[16..80];
        let d = f16::from_le_bytes([q2k[80], q2k[81]]).to_f32();
        let dmin = f16::from_le_bytes([q2k[82], q2k[83]]).to_f32();

        let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
        let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().expect("exact-size slice"));
        let q8 = &q8k_block[4..260];
        let bsums_bytes = &q8k_block[260..292];

        let dall = d * d8;
        let dmin_all = dmin * d8;

        // Sum mins using bsums (matches llama.cpp: summs += bsums[j] * (sc[j] >> 4))
        let mut summs = 0i32;
        for j in 0..16 {
            let bsum = i16::from_le_bytes([bsums_bytes[j * 2], bsums_bytes[j * 2 + 1]]) as i32;
            summs += bsum * (sc[j] >> 4) as i32;
        }

        // Integer dot product per sub-block (matches llama.cpp exactly)
        let mut isum = 0i32;
        let mut is = 0usize;
        let mut q8_offset = 0usize;
        for _n in 0..2 {
            let q = &qs[_n * 32..];
            for shift in (0u8..8).step_by(2) {
                // Sub-block A: 16 elements from q[0..16]
                let sub_scale = (sc[is] & 0x0F) as i32;
                is += 1;
                let mut isuml = 0i32;
                for l in 0..16 {
                    isuml += q8[q8_offset + l] as i8 as i32 * ((q[l] >> shift) & 3) as i32;
                }
                isum += sub_scale * isuml;

                // Sub-block B: 16 elements from q[16..32]
                let sub_scale = (sc[is] & 0x0F) as i32;
                is += 1;
                let mut isuml = 0i32;
                for l in 0..16 {
                    isuml +=
                        q8[q8_offset + 16 + l] as i8 as i32 * ((q[16 + l] >> shift) & 3) as i32;
                }
                isum += sub_scale * isuml;

                q8_offset += 32;
            }
        }

        sumf += dall * isum as f32 - dmin_all * summs as f32;
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::super::quantize_act_q8k::{Q8K_BLOCK_BYTES, quantize_f32_to_q8k};
    use super::*;
    use crate::quant::cpu::kernels::dequant_k_quants;

    #[test]
    fn test_fused_q2k_q8k_vs_f32_dot() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        // Create Q2_K weight block
        let mut weight = [0u8; 84];
        weight[0..16].fill(0x23); // scales: sub_scale=3, sub_min=2
        weight[16..80].fill(0xAA); // qs: q=2 at all shifts
        weight[80..82].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        weight[82..84].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        // Quantize activation to Q8_K
        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q2k_q8k(&act_q8k, &weight, k);

        // Reference: dequant weight, dot with original f32 activation
        let mut dequant_buf = vec![0.0f32; k];
        dequant_k_quants::dequant_q2k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.05 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    #[test]
    fn test_fused_q2k_q8k_large() {
        let k = 4096;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let mut weight = vec![0u8; 84 * 16];
        for blk in 0..16u8 {
            let base = blk as usize * 84;
            weight[base..base + 16].fill(0x12 + blk % 4);
            for i in 16..80 {
                weight[base + i] = ((blk as usize * 17 + i * 31) % 256) as u8;
            }
            weight[base + 80..base + 82]
                .copy_from_slice(&f16::from_f32(0.5 + blk as f32 * 0.03).to_le_bytes());
            weight[base + 82..base + 84]
                .copy_from_slice(&f16::from_f32(0.1 + blk as f32 * 0.01).to_le_bytes());
        }

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * 16];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q2k_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant_k_quants::dequant_q2k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.05 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    /// Verify AVX2 and scalar paths produce identical results when AVX2 is available.
    #[test]
    fn test_fused_q2k_q8k_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                return; // Skip on non-AVX2 hardware
            }

            let k = 512;
            let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).cos()).collect();
            let mut weight = vec![0u8; 84 * 2];
            for blk in 0..2u8 {
                let base = blk as usize * 84;
                for i in 0..16 {
                    weight[base + i] = ((blk as usize * 37 + i * 13) % 256) as u8;
                }
                for i in 16..80 {
                    weight[base + i] = ((blk as usize * 19 + i * 23) % 256) as u8;
                }
                weight[base + 80..base + 82]
                    .copy_from_slice(&f16::from_f32(0.8 + blk as f32 * 0.1).to_le_bytes());
                weight[base + 82..base + 84]
                    .copy_from_slice(&f16::from_f32(0.2 + blk as f32 * 0.05).to_le_bytes());
            }

            let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * 2];
            quantize_f32_to_q8k(&act, &mut act_q8k);

            let scalar = fused_dot_q2k_q8k_scalar(&act_q8k, &weight, k);
            let avx2 = unsafe { fused_dot_q2k_q8k_avx2(&act_q8k, &weight, k) };

            assert!(
                (scalar - avx2).abs() < scalar.abs() * 1e-4 + 1e-4,
                "scalar={scalar}, avx2={avx2}, diff={}",
                (scalar - avx2).abs()
            );
        }
    }
}
