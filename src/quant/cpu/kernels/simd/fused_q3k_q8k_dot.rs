//! AVX2 integer Q3_K × Q8_K dot product
//!
//! Matches llama.cpp's `ggml_vec_dot_q3_K_q8_K` algorithm:
//! unpack 3-bit values to i8 array, integer dot with Q8_K, scale per sub-block.
//!
//! Q3_K block (110 bytes, 256 elements):
//!   [0..32]    hmask (32 bytes, high bit per element)
//!   [32..96]   qs (64 bytes, 2-bit low values packed 4 per byte)
//!   [96..108]  scales (12 bytes, packed 6-bit signed scales)
//!   [108..110] d (f16 scale)
//!
//! Q8_K block (292 bytes, 256 elements):
//!   [0..4]     d (f32), [4..260] qs (i8×256), [260..292] bsums (i16×16)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::super::dequant_k_quants::unpack_q3k_scales;
use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

/// Fused Q3_K × Q8_K dot product using AVX2 maddubs.
///
/// Q3_K 3-bit values are unsigned 0–7; the true value is `q - 4`.  We compute
/// `dot(q_unsigned, q8) - 4 * sum(q8)` and scale per sub-block, matching
/// llama.cpp's integer path.
///
/// # Safety
/// Requires AVX2. `act_q8k` must contain valid Q8_K blocks, `weight` must contain Q3_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_dot_q3k_q8k_avx2(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    unsafe {
        const Q3K_BLOCK_BYTES: usize = 110;
        const Q3K_BLOCK_SIZE: usize = 256;
        let num_blocks = k / Q3K_BLOCK_SIZE;

        let mut sumf = 0.0f32;

        // Constant masks
        let mask2 = _mm256_set1_epi8(0x03);
        let mask4 = _mm256_set1_epi8(0x04);
        let ones_u8 = _mm256_set1_epi8(1i8);

        for b in 0..num_blocks {
            let q3k = &weight[b * Q3K_BLOCK_BYTES..];
            let hm_bytes = &q3k[0..32];
            let qs = &q3k[32..96];
            let sc_packed = &q3k[96..108];
            let d_all = f16::from_le_bytes([q3k[108], q3k[109]]).to_f32();

            let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
            let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap_unchecked());
            let q8_qs = &q8k_block[4..260];

            let d = d_all * d8;

            let scales = unpack_q3k_scales(sc_packed);

            // The structure mirrors the scalar: 2 outer groups × 4 shifts = 8 iterations,
            // each covering 16 elements from sub-block A and 16 from sub-block B (32 total).
            // That yields 16 sub-blocks of 16 elements (= 256 elements total), matching the
            // 16-entry scales array.
            //
            // AVX2: we process 32 elements at a time per (outer_group, shift) step.
            //   - Extract low 2 bits from qs at the given shift.
            //   - Extract the corresponding hmask bit for each element.
            //   - Combine: q_unsigned = low2 | (hm_bit << 2) — gives unsigned 0-7.
            //   - Use maddubs(q_unsigned, q8) for the unsigned×signed dot.
            //   - Correct for the "-4" offset: subtract 4 * maddubs(ones, q8).
            //   - Apply two scales (scale_a for lo128, scale_b for hi128) via madd.
            //   - Accumulate into isum.

            let mut isum = 0i32;
            let mut is: usize = 0; // scale index
            let mut q8_offset: usize = 0;

            // The hmask "m" bit advances with each (shift level).  In the scalar:
            //   m starts at 1, shifts left after each of the 4 shift levels per outer group.
            // So for outer group n and shift index s (0..4):  m_bit = 1 << (n*4 + s)
            // The hmask byte covers 8 elements; the element offset within the 256-element
            // block is: outer n covers elements [n*128 .. (n+1)*128], shift s covers the
            // group of 32 starting at n*128 + s*32 (but in the scalar the 32 elements are
            // the same qs[0..16] and qs[16..32] for sub-blocks A and B respectively, hence
            // all 32 share the same hmask bit position m for that iteration).
            //
            // For the AVX2 path we unpack the hmask bit into a full 32-byte vector.

            for n in 0..2usize {
                let qs_n = &qs[n * 32..]; // 32 bytes, 4-per-byte = 128 elements (4 shifts × 32)
                let raw_qs = _mm256_loadu_si256(qs_n.as_ptr() as *const __m256i);

                for shift in 0..4usize {
                    // Extract 2-bit low values for this shift.
                    // _mm256_srli_epi16 requires a compile-time constant immediate.
                    let shifted_qs = match shift {
                        0 => raw_qs,
                        1 => _mm256_srli_epi16(raw_qs, 2),
                        2 => _mm256_srli_epi16(raw_qs, 4),
                        _ => _mm256_srli_epi16(raw_qs, 6),
                    };
                    let low2_vec = _mm256_and_si256(shifted_qs, mask2);

                    // Expand the hmask bit for this shift level into 32 bytes (0 or 4).
                    // Elements for outer group n at shift s reside at qs indices n*32..n*32+32,
                    // which map to element indices in the block:
                    //   element = n*128 + s*32 + l  (l in 0..32)
                    // The hmask byte for element e is hm[e/8] and bit is (e%8).
                    // In the scalar, m = 1 << (n*4+s); element e uses hm[l] for sub-block A
                    // and hm[16+l] for sub-block B, both with the same m bit.
                    // In the qs byte layout, qs_n[l] holds elements n*128+{0,32,64,96}+l for
                    // shifts 0,1,2,3 respectively.  The hmask byte index for element l at
                    // shift s in outer group n is l (A half) or 16+l (B half), with bit m.
                    //
                    // We pre-expand into a 32-byte array of 0 or 4 (the subtracted offset).
                    let m_bit: u8 = 1u8 << (n * 4 + shift);
                    let mut hmask_arr = [0u8; 32];
                    for l in 0..16 {
                        // Sub-block A: hm byte index = l (same as scalar hm[l] & m)
                        hmask_arr[l] = if hm_bytes[l] & m_bit != 0 { 0 } else { 4 };
                        // Sub-block B: hm byte index = 16+l
                        hmask_arr[16 + l] = if hm_bytes[16 + l] & m_bit != 0 { 0 } else { 4 };
                    }
                    let hmask_vec = _mm256_loadu_si256(hmask_arr.as_ptr() as *const __m256i);

                    // q_unsigned = low2 | (bit_val)  where bit_val encodes the high 3rd bit:
                    //   if hm bit set → high bit = 1 → q_unsigned = low2 | 4
                    //   if hm bit clear → high bit = 0 → q_unsigned = low2
                    //
                    // Equivalently: q_unsigned = low2 | (mask4 & ~hmask_vec_bool)
                    // But hmask_arr already has 0 or 4.  When 0, the element has high bit;
                    // when 4, the element does NOT have high bit but we will subtract 4.
                    // We want q_unsigned to reflect the unsigned 3-bit value:
                    //   hm bit set   → true_val = low2 (high bit 1 in 3-bit = value in [0,3] — wait)
                    //
                    // Re-reading the scalar:
                    //   low2 = (q[l] >> shift) & 3   ← bits [1:0] of 3-bit value
                    //   high_sub = if hm[l] & m != 0 { 0 } else { 4 }
                    //   aux8[a_idx] = low2 as i8 - high_sub
                    //
                    // So:  hm bit SET   → aux8 = low2          (range 0..3)
                    //      hm bit CLEAR → aux8 = low2 - 4      (range -4..-1)
                    //
                    // The true 3-bit unsigned value is: q_u3 = low2 | (hm_bit << 2)
                    //   hm bit SET   → q_u3 = low2 | 4 = low2 + 4   (range 4..7)
                    //   hm bit CLEAR → q_u3 = low2                   (range 0..3)
                    // Then: aux8 = q_u3 - 4
                    //
                    // For maddubs: compute dot(q_u3, q8) - 4 * sum(q8).
                    //   q_u3 = low2 | (hm_bit_set ? 4 : 0)
                    //        = low2 + (hm_bit_set ? 4 : 0)
                    // hmask_arr has 0 when hm_bit is SET and 4 when CLEAR.
                    // So q_u3 = low2 + (4 - hmask_arr[l]).
                    //         = low2 + offset_vec,  where offset_vec = mask4 - hmask_arr
                    //
                    // Then the correction for the -4 offset is: subtract 4*sum(q8).
                    // That's independent of hmask.
                    let offset_vec = _mm256_sub_epi8(mask4, hmask_vec);
                    let q3_u = _mm256_add_epi8(low2_vec, offset_vec);

                    // Load 32 Q8_K activation values
                    let q8_vec =
                        _mm256_loadu_si256(q8_qs.as_ptr().add(q8_offset) as *const __m256i);

                    // dot(q3_u, q8): unsigned × signed → i16
                    let dot16 = _mm256_maddubs_epi16(q3_u, q8_vec);

                    // correction: 4 * sum(q8) = 4 * maddubs(ones, q8)
                    let sum_q8_16 = _mm256_maddubs_epi16(ones_u8, q8_vec);
                    // sum_q8_16 lanes are sums of pairs; each pair contributes 4x
                    // We shift left by 2 (multiply by 4) to get 4*sum(q8) in i16 lanes
                    let corr16 = _mm256_slli_epi16(sum_q8_16, 2);

                    // corrected dot in i16: dot(q3_u, q8) - 4*sum(q8) = dot(q3_u-4, q8)
                    let corrected16 = _mm256_sub_epi16(dot16, corr16);

                    // Apply two scales: scale_a to lo128 (sub-block A), scale_b to hi128 (B)
                    let scale_a = scales[is] as i16;
                    let scale_b = scales[is + 1] as i16;
                    is += 2;

                    let scale_lo = _mm_set1_epi16(scale_a);
                    let scale_hi = _mm_set1_epi16(scale_b);
                    let scale_vec = _mm256_set_m128i(scale_hi, scale_lo);

                    let p = _mm256_madd_epi16(corrected16, scale_vec);

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

            sumf += d * isum as f32;
        }

        sumf
    }
}

/// Dispatch wrapper
#[allow(clippy::needless_range_loop)]
pub fn fused_dot_q3k_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fused_dot_q3k_q8k_avx2(act_q8k, weight, k) };
        }
    }

    fused_dot_q3k_q8k_scalar(act_q8k, weight, k)
}

/// Scalar fallback — matches llama.cpp's `ggml_vec_dot_q3_K_q8_K` exactly.
#[allow(clippy::needless_range_loop)]
fn fused_dot_q3k_q8k_scalar(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    const Q3K_BLOCK_BYTES: usize = 110;
    const Q3K_BLOCK_SIZE: usize = 256;
    let num_blocks = k / Q3K_BLOCK_SIZE;

    // Matches llama.cpp: 8-lane accumulator across blocks
    let mut sums = [0.0f32; 8];
    let mut aux8 = [0i8; 256];

    for b in 0..num_blocks {
        let q3k = &weight[b * Q3K_BLOCK_BYTES..];
        let hm = &q3k[0..32];
        let qs = &q3k[32..96];
        let sc_packed = &q3k[96..108];
        let d_all = f16::from_le_bytes([q3k[108], q3k[109]]).to_f32();

        let q8k_block = &act_q8k[b * Q8K_BLOCK_BYTES..];
        let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().expect("exact-size slice"));
        let q8 = &q8k_block[4..260];

        let d = d_all * d8;

        // Unpack all 256 Q3 values to i8 (matches llama.cpp)
        let mut a_idx = 0;
        let mut m: u8 = 1;
        for n in 0..2 {
            let q = &qs[n * 32..];
            for shift in (0u8..8).step_by(2) {
                for l in 0..32 {
                    let low2 = (q[l] >> shift) & 3;
                    let high_sub = if hm[l] & m != 0 { 0i8 } else { 4i8 };
                    aux8[a_idx] = low2 as i8 - high_sub;
                    a_idx += 1;
                }
                m = m.wrapping_shl(1);
            }
        }

        // Unpack scales
        let scales = unpack_q3k_scales(sc_packed);

        // Integer dot in 8-lane structure (matches llama.cpp)
        let mut aux32 = [0i32; 8];
        let mut q8_off = 0;
        let mut a_off = 0;
        for &sc_val in &scales {
            // 16 sub-blocks of 16 elements each
            let scale = sc_val as i32;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_off + l] as i8 as i32 * aux8[a_off + l] as i32);
            }
            q8_off += 8;
            a_off += 8;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_off + l] as i8 as i32 * aux8[a_off + l] as i32);
            }
            q8_off += 8;
            a_off += 8;
        }

        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }

    sums.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::super::quantize_act_q8k::{Q8K_BLOCK_BYTES, quantize_f32_to_q8k};
    use super::*;
    use crate::quant::cpu::kernels::dequant_k_quants;

    #[test]
    fn test_fused_q3k_q8k_vs_f32_dot() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        // Create Q3_K weight block with known values
        let mut weight = [0u8; 110];
        weight[0..32].fill(0xFF); // hmask: all bits set (subtract 0)
        weight[32..96].fill(0xAA); // qs
        // sc[96..108] = incrementing values
        for i in 0..12 {
            weight[96 + i] = (i as u8 * 17 + 5) % 64;
        }
        weight[108..110].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q3k_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant_k_quants::dequant_q3k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.05 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    #[test]
    fn test_fused_q3k_q8k_large() {
        let k = 4096;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let mut weight = vec![0u8; 110 * 16];
        for blk in 0..16u8 {
            let base = blk as usize * 110;
            weight[base..base + 32].fill(0xF0 ^ blk);
            for i in 32..96 {
                weight[base + i] = ((blk as usize * 13 + i * 37) % 256) as u8;
            }
            for i in 96..108 {
                weight[base + i] = ((blk as usize * 7 + i * 11) % 64) as u8;
            }
            weight[base + 108..base + 110]
                .copy_from_slice(&f16::from_f32(0.3 + blk as f32 * 0.02).to_le_bytes());
        }

        let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * 16];
        quantize_f32_to_q8k(&act, &mut act_q8k);

        let result = fused_dot_q3k_q8k(&act_q8k, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant_k_quants::dequant_q3k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - reference).abs() < reference.abs() * 0.05 + 1.0,
            "q8k={result}, f32_ref={reference}, diff={}",
            (result - reference).abs()
        );
    }

    /// Verify AVX2 and scalar paths produce identical results when AVX2 is available.
    #[test]
    fn test_fused_q3k_q8k_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                return; // Skip on non-AVX2 hardware
            }

            let k = 512;
            let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.007).cos()).collect();
            let mut weight = vec![0u8; 110 * 2];
            for blk in 0..2u8 {
                let base = blk as usize * 110;
                // hmask: mix of set and clear bits
                for i in 0..32 {
                    weight[base + i] = ((blk as usize * 43 + i * 11) % 256) as u8;
                }
                for i in 32..96 {
                    weight[base + i] = ((blk as usize * 19 + i * 23) % 256) as u8;
                }
                for i in 96..108 {
                    weight[base + i] = ((blk as usize * 7 + i * 11) % 64) as u8;
                }
                weight[base + 108..base + 110]
                    .copy_from_slice(&f16::from_f32(0.6 + blk as f32 * 0.15).to_le_bytes());
            }

            let mut act_q8k = vec![0u8; Q8K_BLOCK_BYTES * 2];
            quantize_f32_to_q8k(&act, &mut act_q8k);

            let scalar = fused_dot_q3k_q8k_scalar(&act_q8k, &weight, k);
            let avx2 = unsafe { fused_dot_q3k_q8k_avx2(&act_q8k, &weight, k) };

            assert!(
                (scalar - avx2).abs() < scalar.abs() * 1e-4 + 1e-4,
                "scalar={scalar}, avx2={avx2}, diff={}",
                (scalar - avx2).abs()
            );
        }
    }
}
