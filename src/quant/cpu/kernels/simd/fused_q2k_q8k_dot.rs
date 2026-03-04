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

use half::f16;

use super::quantize_act_q8k::Q8K_BLOCK_BYTES;

/// Fused Q2_K × Q8_K dot product — scalar implementation matching llama.cpp exactly.
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

/// Dispatch wrapper
pub fn fused_dot_q2k_q8k(act_q8k: &[u8], weight: &[u8], k: usize) -> f32 {
    // TODO: AVX2 version with maddubs
    fused_dot_q2k_q8k_scalar(act_q8k, weight, k)
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
}
