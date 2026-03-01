//! AVX2 fused dequant+dot for Q6_K format
//!
//! Dequantizes Q6_K blocks and accumulates dot product with activation in a single
//! SIMD pass — no intermediate f32 buffer needed.
//!
//! Q6_K block layout (256 elements, 210 bytes):
//!   [0..128]   ql — low 4 bits of 6-bit values
//!   [128..192]  qh — high 2 bits of 6-bit values
//!   [192..208]  sc — 16 × i8 scales
//!   [208..210]  d  — f16 overall scale
//!
//! 2 halves of 128 elements. Each half has 4 groups of 32, sharing scale indices.
//! Per element: q6 = (ql_nibble | (qh_bits << 4)) - 32
//! dequant(i) = d * sc[idx] * q6

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

/// Fused dequant+dot for Q6_K using AVX2+FMA.
///
/// Computes: sum_i(act[i] * dequant(weight[i])) for a full weight row.
///
/// # Safety
/// Requires AVX2+FMA. Caller must ensure act.len() >= k and blocks covers k/256 blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_dot_q6k_avx2(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    unsafe {
        const BLOCK_SIZE: usize = 256;
        const BLOCK_BYTES: usize = 210;
        let num_blocks = k / BLOCK_SIZE;

        let mut total_acc = _mm256_setzero_ps();
        let bias32 = _mm256_set1_epi32(32);

        for b in 0..num_blocks {
            let block = &blocks[b * BLOCK_BYTES..];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
            let act_block = &act[b * BLOCK_SIZE..];

            // Process in two halves of 128 elements
            for n in 0..2u32 {
                let y_base = (n as usize) * 128;
                let ql_base = (n as usize) * 64;
                let qh_base = (n as usize) * 32;
                let sc_base = (n as usize) * 8;

                for g in 0..4 {
                    let l_start = g * 8;

                    let ql_ptr0 = ql.as_ptr().add(ql_base + l_start);
                    let ql_raw0 = _mm_loadl_epi64(ql_ptr0 as *const __m128i);
                    let ql_32_0 = _mm256_cvtepu8_epi32(ql_raw0);
                    let ql_lo0 = _mm256_and_si256(ql_32_0, _mm256_set1_epi32(0x0F));
                    let ql_hi0 = _mm256_srli_epi32(ql_32_0, 4);

                    let ql_ptr1 = ql.as_ptr().add(ql_base + 32 + l_start);
                    let ql_raw1 = _mm_loadl_epi64(ql_ptr1 as *const __m128i);
                    let ql_32_1 = _mm256_cvtepu8_epi32(ql_raw1);
                    let ql_lo1 = _mm256_and_si256(ql_32_1, _mm256_set1_epi32(0x0F));
                    let ql_hi1 = _mm256_srli_epi32(ql_32_1, 4);

                    let qh_ptr = qh.as_ptr().add(qh_base + l_start);
                    let qh_raw = _mm_loadl_epi64(qh_ptr as *const __m128i);
                    let qh_32 = _mm256_cvtepu8_epi32(qh_raw);

                    let qh_bits0 =
                        _mm256_slli_epi32(_mm256_and_si256(qh_32, _mm256_set1_epi32(0x03)), 4);
                    let qh_bits1 = _mm256_slli_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(qh_32, 2), _mm256_set1_epi32(0x03)),
                        4,
                    );
                    let qh_bits2 = _mm256_slli_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(qh_32, 4), _mm256_set1_epi32(0x03)),
                        4,
                    );
                    let qh_bits3 = _mm256_slli_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(qh_32, 6), _mm256_set1_epi32(0x03)),
                        4,
                    );

                    let q0 = _mm256_sub_epi32(_mm256_or_si256(ql_lo0, qh_bits0), bias32);
                    let q1 = _mm256_sub_epi32(_mm256_or_si256(ql_lo1, qh_bits1), bias32);
                    let q2 = _mm256_sub_epi32(_mm256_or_si256(ql_hi0, qh_bits2), bias32);
                    let q3 = _mm256_sub_epi32(_mm256_or_si256(ql_hi1, qh_bits3), bias32);

                    let is = g / 2;

                    let sc_ptr = sc.as_ptr().add(sc_base) as *const i8;
                    let s0 = d * (*sc_ptr.add(is) as f32);
                    let s1 = d * (*sc_ptr.add(is + 2) as f32);
                    let s2 = d * (*sc_ptr.add(is + 4) as f32);
                    let s3 = d * (*sc_ptr.add(is + 6) as f32);

                    let s0_vec = _mm256_set1_ps(s0);
                    let s1_vec = _mm256_set1_ps(s1);
                    let s2_vec = _mm256_set1_ps(s2);
                    let s3_vec = _mm256_set1_ps(s3);

                    let q0_f32 = _mm256_cvtepi32_ps(q0);
                    let q1_f32 = _mm256_cvtepi32_ps(q1);
                    let q2_f32 = _mm256_cvtepi32_ps(q2);
                    let q3_f32 = _mm256_cvtepi32_ps(q3);

                    let a0 = _mm256_loadu_ps(act_block.as_ptr().add(y_base + l_start));
                    let a1 = _mm256_loadu_ps(act_block.as_ptr().add(y_base + l_start + 32));
                    let a2 = _mm256_loadu_ps(act_block.as_ptr().add(y_base + l_start + 64));
                    let a3 = _mm256_loadu_ps(act_block.as_ptr().add(y_base + l_start + 96));

                    let w0 = _mm256_mul_ps(s0_vec, q0_f32);
                    total_acc = _mm256_fmadd_ps(a0, w0, total_acc);

                    let w1 = _mm256_mul_ps(s1_vec, q1_f32);
                    total_acc = _mm256_fmadd_ps(a1, w1, total_acc);

                    let w2 = _mm256_mul_ps(s2_vec, q2_f32);
                    total_acc = _mm256_fmadd_ps(a2, w2, total_acc);

                    let w3 = _mm256_mul_ps(s3_vec, q3_f32);
                    total_acc = _mm256_fmadd_ps(a3, w3, total_acc);
                }
            }
        }

        super::dot_f32::hsum_f32_avx2(total_acc)
    }
}

/// Dispatch wrapper: uses AVX2 if available, falls back to scalar
pub fn fused_dot_q6k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { fused_dot_q6k_avx2(act, blocks, k) };
        }
    }

    // Scalar fallback
    super::super::fused_dot::fused_dot_row(act, blocks, k, crate::quant::QuantFormat::Q6K)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    #[test]
    fn test_fused_q6k_avx2_vs_dequant() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let mut block = [0u8; 210];
        // d = 0.5
        block[208..210].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        // scales all = 1 (as i8)
        block[192..208].fill(1);
        // ql and qh all zeros → q = 0 - 32 = -32

        let fused = fused_dot_q6k(&act, &block, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q6k(&block, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_q6k_avx2_multi_block() {
        let k = 512;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut weight = vec![0u8; 210 * 2];
        for blk in 0..2 {
            let base = blk * 210;
            weight[base + 208..base + 210].copy_from_slice(&f16::from_f32(1.5).to_le_bytes());
            // Varied scales
            for i in 0..16 {
                weight[base + 192 + i] = ((i as i8 % 5) + 1) as u8;
            }
            // Varied ql and qh
            for i in 0..128 {
                weight[base + i] = ((blk * 17 + i * 31) % 256) as u8;
            }
            for i in 0..64 {
                weight[base + 128 + i] = ((blk * 13 + i * 37) % 256) as u8;
            }
        }

        let fused = fused_dot_q6k(&act, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q6k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-1,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_q6k_avx2_large() {
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

        let fused = fused_dot_q6k(&act, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q6k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-1,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }
}
