//! Fused dequant+dot for Q2_K format
//!
//! Per-block accumulation avoids precision loss from full-row dequant+dot.
//!
//! Q2_K block (84 bytes, 256 elements):
//!   [0..16]  scales (low 4 = sub-scale, high 4 = sub-min)
//!   [16..80] qs (64 bytes, 2-bit packed 4/byte)
//!   [80..82] d (f16), [82..84] dmin (f16)

use half::f16;

/// Fused dequant+dot for Q2_K — per-block f32 accumulation.
pub fn fused_dot_q2k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 84;
    let num_blocks = k / BLOCK_SIZE;

    let mut sumf = 0.0f32;

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let sc = &block[0..16];
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
        let act_block = &act[b * BLOCK_SIZE..][..BLOCK_SIZE];

        let mut scale_sum = 0.0f32;
        let mut min_sum = 0.0f32;
        let mut y = 0;
        let mut is = 0;
        let mut q_offset = 0;
        for _n in 0..2 {
            let q = &qs[q_offset..];
            for shift in (0u8..8).step_by(2) {
                let sub_scale = (sc[is] & 0x0F) as f32;
                let sub_min = (sc[is] >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    let qval = ((q[l] >> shift) & 3) as f32;
                    scale_sum += act_block[y] * sub_scale * qval;
                    min_sum += act_block[y] * sub_min;
                    y += 1;
                }
                let sub_scale = (sc[is] & 0x0F) as f32;
                let sub_min = (sc[is] >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    let qval = ((q[16 + l] >> shift) & 3) as f32;
                    scale_sum += act_block[y] * sub_scale * qval;
                    min_sum += act_block[y] * sub_min;
                    y += 1;
                }
            }
            q_offset += 32;
        }
        sumf += d * scale_sum - dmin * min_sum;
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant_k_quants;

    #[test]
    fn test_fused_q2k_vs_dequant() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let mut block = [0u8; 84];
        block[0..16].fill(0x23);
        block[16..80].fill(0xAA);
        block[80..82].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[82..84].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        let fused = fused_dot_q2k(&act, &block, k);

        let mut dequant_buf = vec![0.0f32; k];
        dequant_k_quants::dequant_q2k(&block, &mut dequant_buf);
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-4 + 1e-2,
            "fused={fused}, reference={reference}",
        );
    }
}
