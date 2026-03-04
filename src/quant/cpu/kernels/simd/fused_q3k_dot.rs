//! Fused dequant+dot for Q3_K format
//!
//! Per-block accumulation avoids precision loss from full-row dequant+dot.
//!
//! Q3_K block (110 bytes, 256 elements):
//!   [0..32]    hmask, [32..96] qs, [96..108] scales, [108..110] d (f16)

use half::f16;

use super::super::dequant_k_quants::unpack_q3k_scales;

/// Fused dequant+dot for Q3_K — per-block f32 accumulation.
#[allow(clippy::needless_range_loop)]
pub fn fused_dot_q3k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let num_blocks = k / BLOCK_SIZE;

    let mut sumf = 0.0f32;

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let hm = &block[0..32];
        let qs = &block[32..96];
        let sc_packed = &block[96..108];
        let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();
        let act_block = &act[b * BLOCK_SIZE..][..BLOCK_SIZE];

        let scales = unpack_q3k_scales(sc_packed);

        let mut block_sum = 0.0f32;
        let mut y = 0;
        let mut is = 0;
        let mut m: u8 = 1;
        let mut q_offset = 0;
        for _n in 0..2 {
            let q = &qs[q_offset..];
            for shift in (0u8..8).step_by(2) {
                let dl = scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let low2 = (q[l] >> shift) & 3;
                    let high_sub = if hm[l] & m != 0 { 0i8 } else { 4i8 };
                    let qval = (low2 as i8 - high_sub) as f32;
                    block_sum += act_block[y] * dl * qval;
                    y += 1;
                }
                let dl = scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let low2 = (q[16 + l] >> shift) & 3;
                    let high_sub = if hm[16 + l] & m != 0 { 0i8 } else { 4i8 };
                    let qval = (low2 as i8 - high_sub) as f32;
                    block_sum += act_block[y] * dl * qval;
                    y += 1;
                }
                m = m.wrapping_shl(1);
            }
            q_offset += 32;
        }
        sumf += d_all * block_sum;
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant_k_quants;

    #[test]
    fn test_fused_q3k_vs_dequant() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let mut block = [0u8; 110];
        block[0..32].fill(0xFF);
        block[32..96].fill(0xAA);
        for (i, b) in block[96..108].iter_mut().enumerate() {
            *b = (((i + 96) * 17 + 5) % 64) as u8;
        }
        block[108..110].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        let fused = fused_dot_q3k(&act, &block, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant_k_quants::dequant_q3k(&block, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}",
        );
    }
}
