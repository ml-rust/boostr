//! Fused dequant+dot kernels for quantized matmul
//!
//! Instead of dequantizing an entire weight row to f32 then computing a dot product
//! (2 passes over K elements), these kernels dequantize block-by-block and accumulate
//! the dot product in registers — single pass, no intermediate buffer.

use half::f16;

use crate::quant::QuantFormat;

use super::dequant_k_quants::unpack_q4k_q5k_scales;

/// Fused dequant+dot for a single weight row against one activation row.
/// Returns the dot product without allocating an intermediate buffer.
pub fn fused_dot_row(
    act_row: &[f32],
    weight_row_bytes: &[u8],
    k: usize,
    format: QuantFormat,
) -> f32 {
    match format {
        QuantFormat::Q4_0 => fused_dot_q4_0(act_row, weight_row_bytes, k),
        QuantFormat::Q8_0 => fused_dot_q8_0(act_row, weight_row_bytes, k),
        QuantFormat::Q4K => fused_dot_q4k(act_row, weight_row_bytes, k),
        QuantFormat::Q6K => fused_dot_q6k(act_row, weight_row_bytes, k),
        _ => 0.0, // caller should validate
    }
}

/// Fused dequant+dot for Q4_0 (32-element blocks, 18 bytes each)
fn fused_dot_q4_0(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let num_blocks = k / BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        let act_block = &act[b * BLOCK_SIZE..][..BLOCK_SIZE];

        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = qs[i];
            let low = (byte & 0x0F) as i8 - 8;
            let high = ((byte >> 4) & 0x0F) as i8 - 8;
            block_sum += act_block[i * 2] * low as f32;
            block_sum += act_block[i * 2 + 1] * high as f32;
        }
        sum += d * block_sum;
    }
    sum
}

/// Fused dequant+dot for Q8_0 (32-element blocks, 34 bytes each)
fn fused_dot_q8_0(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let num_blocks = k / BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let act_block = &act[b * BLOCK_SIZE..][..BLOCK_SIZE];

        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += act_block[i] * qs[i] as i8 as f32;
        }
        sum += d * block_sum;
    }
    sum
}

/// Fused dequant+dot for Q4_K (256-element blocks, 144 bytes each)
///
/// This is the hot path for Q4_K_M models (e.g. Mistral 7B).
/// Processes 8 sub-blocks of 32 elements per block.
fn fused_dot_q4k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let num_blocks = k / BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let sc = &block[4..16];
        let qs = &block[16..144];
        let act_block = &act[b * BLOCK_SIZE..][..BLOCK_SIZE];

        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        // For each sub-block of 32 elements, accumulate:
        //   sum += d * scale * (act · q_nibbles) - dmin * min * sum(act)
        // This avoids materializing the dequantized values.
        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;

            let chunk = j / 2;
            let is_high = j % 2 == 1;
            let qs_base = chunk * 32;

            let act_sub = &act_block[j * 32..][..32];
            let mut dot_sum = 0.0f32;
            let mut act_sum = 0.0f32;

            for l in 0..32 {
                let q = if is_high {
                    ((qs[qs_base + l] >> 4) & 0x0F) as f32
                } else {
                    (qs[qs_base + l] & 0x0F) as f32
                };
                dot_sum += act_sub[l] * q;
                act_sum += act_sub[l];
            }
            sum += dl * dot_sum - ml * act_sum;
        }
    }
    sum
}

/// Fused dequant+dot for Q6_K (256-element blocks, 210 bytes each)
fn fused_dot_q6k(act: &[f32], blocks: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let num_blocks = k / BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc: &[i8] = unsafe { std::slice::from_raw_parts(block[192..208].as_ptr().cast(), 16) };
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let act_block = &act[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Process in two halves of 128 elements
        for n in 0..2 {
            let y_base = n * 128;
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;

            for l in 0..32 {
                let is = l / 16;

                let q1 = ((ql[ql_base + l] & 0x0F) | ((qh[qh_base + l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4))
                    as i8
                    - 32;
                let q3 =
                    ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4))
                    as i8
                    - 32;

                sum += d * sc[sc_base + is] as f32 * q1 as f32 * act_block[y_base + l];
                sum += d * sc[sc_base + is + 2] as f32 * q2 as f32 * act_block[y_base + l + 32];
                sum += d * sc[sc_base + is + 4] as f32 * q3 as f32 * act_block[y_base + l + 64];
                sum += d * sc[sc_base + is + 6] as f32 * q4 as f32 * act_block[y_base + l + 96];
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant;

    /// Helper: compare fused dot against dequant-then-dot for a given format
    fn verify_fused_vs_dequant(act: &[f32], weight_bytes: &[u8], k: usize, format: QuantFormat) {
        let fused = fused_dot_row(act, weight_bytes, k, format);

        // Reference: dequant then dot
        let mut dequant_buf = vec![0.0f32; k];
        match format {
            QuantFormat::Q4_0 => dequant::dequant_q4_0(weight_bytes, &mut dequant_buf),
            QuantFormat::Q8_0 => dequant::dequant_q8_0(weight_bytes, &mut dequant_buf),
            QuantFormat::Q4K => dequant::dequant_q4k(weight_bytes, &mut dequant_buf),
            QuantFormat::Q6K => dequant::dequant_q6k(weight_bytes, &mut dequant_buf),
            _ => panic!("unsupported"),
        }
        let reference: f32 = act.iter().zip(dequant_buf.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < reference.abs() * 1e-5 + 1e-3,
            "fused={fused}, reference={reference}, diff={}",
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_q4_0() {
        let k = 32;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[2..18].fill(0x99);
        verify_fused_vs_dequant(&act, &block, k, QuantFormat::Q4_0);
    }

    #[test]
    fn test_fused_q8_0() {
        let k = 32;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2..34].fill(4);
        verify_fused_vs_dequant(&act, &block, k, QuantFormat::Q8_0);
    }

    #[test]
    fn test_fused_q4k() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[4..8].fill(0x01); // scales[0..4] = 1
        block[8..12].fill(0x01); // mins[0..4] = 1
        block[16..144].fill(0x55); // nibble=5 for both halves
        verify_fused_vs_dequant(&act, &block, k, QuantFormat::Q4K);
    }

    #[test]
    fn test_fused_q6k() {
        let k = 256;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let mut block = [0u8; 210];
        block[208..210].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        // scales all = 1 (as i8)
        block[192..208].fill(1);
        // ql and qh all zeros → q = 0 - 32 = -32 for all
        verify_fused_vs_dequant(&act, &block, k, QuantFormat::Q6K);
    }

    #[test]
    fn test_fused_q4k_multi_block() {
        // Test with multiple blocks (K=512 = 2 Q4K blocks)
        let k = 512;
        let act: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut weight = vec![0u8; 144 * 2];
        for blk in 0..2 {
            let base = blk * 144;
            weight[base..base + 2].copy_from_slice(&f16::from_f32(1.5).to_le_bytes());
            weight[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.3).to_le_bytes());
            weight[base + 4..base + 8].fill(0x02);
            weight[base + 8..base + 12].fill(0x01);
            weight[base + 16..base + 144].fill(0x37);
        }
        verify_fused_vs_dequant(&act, &weight, k, QuantFormat::Q4K);
    }
}
