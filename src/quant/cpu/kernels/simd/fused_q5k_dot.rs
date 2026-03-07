//! AVX2 fused dequant+dot for Q5_K format
//!
//! Q5_K block layout (256 elements, 176 bytes):
//!   [0..2]   d (f16 scale)
//!   [2..4]   dmin (f16 minimum)
//!   [4..16]  sc (12-byte packed 6-bit scales+mins for 8 sub-blocks)
//!   [16..48] qh (32 bytes, 1 high bit per element)
//!   [48..176] qs (128 bytes of 4-bit low nibbles, 2 per byte)
//!
//! 8 sub-blocks of 32 elements. Value = low4 | (high1 << 4), range [0, 31].
//! dequant(i) = d * scale[j] * value - dmin * min[j]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use super::super::dequant_k_quants::unpack_q4k_q5k_scales;

/// Fused dequant+dot for Q5_K using AVX2+FMA.
///
/// # Safety
/// Requires AVX2+FMA. Caller must ensure act.len() >= k and blocks covers k/256 blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_dot_q5k_avx2(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let num_blocks = k / BLOCK_SIZE;

    let mut total_acc = _mm256_setzero_ps();

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let sc = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let act_block = &act[b * BLOCK_SIZE..];

        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        // Process 8 sub-blocks of 32 elements each
        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;
            let act_sub = &act_block[j * 32..];

            let dl_vec = _mm256_set1_ps(dl);
            let ml_vec = _mm256_set1_ps(ml);

            // Process 32 elements in 4 groups of 8
            for g in 0..4 {
                let l_base = g * 8;
                let idx_base = j * 32 + l_base; // global element index within block

                unsafe {
                    // Load 8 bytes of qs, compute qs_idx = j*16 + (l_base+i)/2
                    // For each element l in sub-block j:
                    //   qs_idx = j*16 + l/2
                    //   low4 = (l%2==0) ? qs[qs_idx] & 0xF : qs[qs_idx] >> 4
                    //   high1 = (qh[idx/8] >> (idx%8)) & 1
                    //   value = low4 | (high1 << 4)

                    // Build 8 values manually (SIMD gather would be slower for this pattern)
                    let mut vals = [0i32; 8];
                    for (i, val) in vals.iter_mut().enumerate() {
                        let l = l_base + i;
                        let idx = idx_base + i;
                        let qs_idx = j * 16 + l / 2;
                        let low4 = if l % 2 == 0 {
                            (qs[qs_idx] & 0x0F) as i32
                        } else {
                            ((qs[qs_idx] >> 4) & 0x0F) as i32
                        };
                        let high1 = ((qh[idx / 8] >> (idx % 8)) & 1) as i32;
                        *val = low4 | (high1 << 4);
                    }

                    let q_i32 = _mm256_loadu_si256(vals.as_ptr() as *const __m256i);
                    let q_f32 = _mm256_cvtepi32_ps(q_i32);

                    let a = _mm256_loadu_ps(act_sub.as_ptr().add(l_base));

                    // a * (dl * q - ml) = dl * (a * q) - ml * a
                    let aq = _mm256_mul_ps(a, q_f32);
                    total_acc = _mm256_fmadd_ps(dl_vec, aq, total_acc);
                    total_acc = _mm256_fnmadd_ps(ml_vec, a, total_acc);
                }
            }
        }
    }

    unsafe { super::dot_f32::hsum_f32_avx2(total_acc) }
}

/// Dispatch wrapper: uses AVX2 if available, falls back to scalar
pub fn fused_dot_q5k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { fused_dot_q5k_avx2(act, blocks, k) };
        }
    }

    super::super::fused_dot::fused_dot_row(act, blocks, k, crate::quant::QuantFormat::Q5K)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    #[test]
    fn test_fused_q5k_avx2_vs_dequant() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let mut block = [0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[4..8].fill(0x03);
        block[8..12].fill(0x02);
        block[12..16].fill(0x10);
        block[16..48].fill(0xAA); // qh: alternating high bits
        block[48..176].fill(0x73); // qs: nibbles 3 and 7

        let fused = fused_dot_q5k(&act, &block, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q5k(&block, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_q5k_avx2_multi_block() {
        let k = 512;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut weight = vec![0u8; 176 * 2];
        for blk in 0..2 {
            let base = blk * 176;
            weight[base..base + 2].copy_from_slice(&f16::from_f32(1.5).to_le_bytes());
            weight[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.3).to_le_bytes());
            weight[base + 4..base + 8].fill(0x05);
            weight[base + 8..base + 12].fill(0x01);
            weight[base + 16..base + 48].fill(0x55);
            weight[base + 48..base + 176].fill(0xA5);
        }

        let fused = fused_dot_q5k(&act, &weight, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant::dequant_q5k(&weight, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }
}
