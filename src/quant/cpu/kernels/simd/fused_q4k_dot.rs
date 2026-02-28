//! AVX2 fused dequant+dot for Q4_K format
//!
//! Dequantizes Q4_K blocks and accumulates dot product with activation in a single
//! SIMD pass — no intermediate f32 buffer needed. This is the hot kernel for
//! Q4_K_M model inference (e.g. Mistral 7B).
//!
//! Q4_K block layout (256 elements, 144 bytes):
//!   [0..2]   d (f16 scale)
//!   [2..4]   dmin (f16 minimum)
//!   [4..16]  sc (12-byte packed 6-bit scales+mins for 8 sub-blocks)
//!   [16..144] qs (128 bytes of 4-bit quantized values, 2 per byte)
//!
//! 8 sub-blocks of 32 elements. Sub-blocks 0,1 share qs[0..32]; 2,3 share qs[32..64]; etc.
//! Even sub-blocks use low nibble, odd sub-blocks use high nibble.
//! dequant(i) = d * scale[j] * nibble - dmin * min[j]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::super::dequant_k_quants::unpack_q4k_q5k_scales;

/// Fused dequant+dot for Q4_K using AVX2+FMA.
///
/// Computes: sum_i(act[i] * dequant(weight[i])) for a full weight row.
///
/// # Safety
/// Requires AVX2+FMA. Caller must ensure act.len() >= k and blocks covers k/256 blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_dot_q4k_avx2(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let num_blocks = k / BLOCK_SIZE;

    let mut total_acc = _mm256_setzero_ps();

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let sc = &block[4..16];
        let qs = &block[16..144];
        let act_block = &act[b * BLOCK_SIZE..];

        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        // Process 8 sub-blocks of 32 elements each
        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;

            let chunk = j / 2;
            let is_high = j % 2 == 1;
            let qs_base = chunk * 32;

            let act_sub = &act_block[j * 32..];

            // Process 32 elements in 4 groups of 8
            let dl_vec = _mm256_set1_ps(dl);
            let ml_vec = _mm256_set1_ps(ml);

            // Q4_K: sub-block j uses qs_base, and:
            //   even j: nibble = qs[qs_base + l] & 0x0F
            //   odd j:  nibble = qs[qs_base + l] >> 4
            // where l goes 0..32

            // Process 8 elements at a time (load 8 bytes of qs, extract 8 nibbles)
            for g in 0..4 {
                let l_base = g * 8;

                unsafe {
                    // Load 8 bytes of quantized data
                    let qs_ptr = qs.as_ptr().add(qs_base + l_base);
                    // Load 8 bytes into low 64 bits, zero-extend to 32-bit integers
                    let raw = _mm_loadl_epi64(qs_ptr as *const __m128i);
                    let raw256 = _mm256_cvtepu8_epi32(raw);

                    // Extract nibbles
                    let nibbles = if is_high {
                        _mm256_srli_epi32(raw256, 4)
                    } else {
                        raw256
                    };
                    let nibbles = _mm256_and_si256(nibbles, _mm256_set1_epi32(0x0F));

                    // Convert to f32
                    let q_f32 = _mm256_cvtepi32_ps(nibbles);

                    // Load 8 activation values
                    let a = _mm256_loadu_ps(act_sub.as_ptr().add(l_base));

                    // Accumulate: a * (dl * q - ml) = dl * (a * q) - ml * a
                    let aq = _mm256_mul_ps(a, q_f32);
                    total_acc = _mm256_fmadd_ps(dl_vec, aq, total_acc);
                    total_acc = _mm256_fnmadd_ps(ml_vec, a, total_acc);
                }
            }
        }
    }

    // Horizontal sum
    unsafe { super::dot_f32::hsum_f32_avx2(total_acc) }
}

/// Dispatch wrapper: uses AVX2 if available, falls back to scalar
pub fn fused_dot_q4k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { fused_dot_q4k_avx2(act, blocks, k) };
        }
    }

    // Scalar fallback
    super::super::fused_dot::fused_dot_row(act, blocks, k, crate::quant::QuantFormat::Q4K)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    #[test]
    fn test_fused_q4k_avx2_vs_dequant() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[4..8].fill(0x03);
        block[8..12].fill(0x02);
        block[12..16].fill(0x10);
        block[16..144].fill(0x73);

        let fused = fused_dot_q4k(&act, &block, k);

        // Reference: dequant then dot
        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q4k(&block, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_q4k_avx2_multi_block() {
        let k = 512;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut weight = vec![0u8; 144 * 2];
        for blk in 0..2 {
            let base = blk * 144;
            weight[base..base + 2].copy_from_slice(&f16::from_f32(1.5).to_le_bytes());
            weight[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.3).to_le_bytes());
            weight[base + 4..base + 8].fill(0x05);
            weight[base + 8..base + 12].fill(0x01);
            weight[base + 16..base + 144].fill(0xA5);
        }

        let fused = fused_dot_q4k(&act, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q4k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_q4k_avx2_large() {
        // Test with K=4096 (typical for Mistral 7B)
        let k = 4096;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let mut weight = vec![0u8; 144 * 16]; // 16 blocks of 256
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

        let fused = fused_dot_q4k(&act, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q4k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-1,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }
}
