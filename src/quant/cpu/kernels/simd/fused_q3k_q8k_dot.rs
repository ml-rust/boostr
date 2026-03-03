//! Integer Q3_K × Q8_K dot product
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

use half::f16;

use super::super::dequant_k_quants::unpack_q3k_scales;
use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

/// Fused Q3_K × Q8_K dot product — scalar matching llama.cpp.
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
        let d8 = f32::from_le_bytes(q8k_block[0..4].try_into().unwrap());
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
        for j in 0..16 {
            // 16 sub-blocks of 16 elements each
            let scale = scales[j] as i32;
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

/// Dispatch wrapper
pub fn fused_dot_q3k_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    // TODO: AVX2 version
    fused_dot_q3k_q8k_scalar(act_q8k, weight, k)
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
}
