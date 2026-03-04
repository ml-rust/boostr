//! CPU dequantization kernels for IQ3 formats
//!
//! IQ3_XXS, IQ3_S
use half::f16;

/// Dequantize IQ3_XXS blocks to f32
///
/// IQ3_XXS: 256 elements, 98 bytes/block
/// Layout: f16 d (2B) + 96 bytes packed data
/// 8 groups of 32 values, each group encoded in 12 bytes
pub fn dequant_iq3_xxs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 98;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..98]; // 96 bytes

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 8 groups, 12 bytes each (8B grid data + 4B signs/scales)
        for group in 0..8 {
            let gdata = &qs[group * 12..(group + 1) * 12];

            // First 8 bytes: 4 grid indices (16 bits each)
            // Last 4 bytes: sign bits + scale
            let signs = u32::from_le_bytes([gdata[8], gdata[9], gdata[10], gdata[11]]);
            let sub_scale_bits = (signs >> 28) & 0x0F;
            let sub_scale = d * (1.0 + sub_scale_bits as f32);

            let group_out = &mut out[group * 32..(group + 1) * 32];

            for sub in 0..4 {
                let grid_lo = gdata[sub * 2] as u16;
                let grid_hi = gdata[sub * 2 + 1] as u16;
                let grid_idx = (grid_lo | (grid_hi << 8)) as usize;

                for k in 0..8 {
                    let val = ((grid_idx >> (k * 2)) & 0x03) as f32 + 1.0;
                    let sign_bit = (signs >> (sub * 8 + k)) & 1;
                    let sign = if sign_bit != 0 { -1.0f32 } else { 1.0f32 };
                    group_out[sub * 8 + k] = sub_scale * val * sign;
                }
            }
        }
    }
}

/// Dequantize IQ3_S blocks to f32
///
/// IQ3_S: 256 elements, 110 bytes/block
/// Layout: f16 d (2B) + qs (32B) + qh (4B) + signs (32B) + scales (8B)
#[allow(clippy::needless_range_loop)]
pub fn dequant_iq3_s(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 110;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34]; // 32 bytes of 3-bit grid indices
        let qh = &block[34..38]; // 4 bytes of high bits
        let signs = &block[38..70]; // 32 bytes of sign bits
        let scales = &block[70..78]; // 8 bytes of sub-block scales

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 8 sub-blocks of 32 values
        for sb in 0..8 {
            let scale_byte = scales[sb];
            let sub_scale = d * (1.0 + (scale_byte & 0x0F) as f32);

            let sub_out = &mut out[sb * 32..(sb + 1) * 32];

            for k in 0..32 {
                let byte_idx = sb * 4 + k / 8;
                let bit_pos = k % 8;
                let q3 = if byte_idx < 32 {
                    ((qs[byte_idx] >> (bit_pos % 4 * 2)) & 0x03) as f32
                } else {
                    0.0
                };
                // High bit
                let qh_byte_idx = (sb * 32 + k) / 8;
                let qh_bit = if qh_byte_idx < 4 {
                    ((qh[qh_byte_idx] >> ((sb * 32 + k) % 8)) & 1) as f32
                } else {
                    0.0
                };
                let val = q3 + qh_bit * 4.0 + 1.0;

                let sign_byte_idx = sb * 4 + k / 8;
                let sign_byte = if sign_byte_idx < 32 {
                    signs[sign_byte_idx]
                } else {
                    0
                };
                let sign = if (sign_byte >> (k % 8)) & 1 != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };

                sub_out[k] = sub_scale * val * sign;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_iq3_xxs_zeros() {
        let block = [0u8; 98];
        let mut output = [0.0f32; 256];
        dequant_iq3_xxs(&block, &mut output);
        // With all-zero input, scale bits = 0 → sub_scale = d * 1.0 = 0
        // all values should be 0
        for &v in &output {
            assert!(v.abs() < 1e-5, "expected 0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_iq3_s_zeros() {
        let block = [0u8; 110];
        let mut output = [0.0f32; 256];
        dequant_iq3_s(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5, "expected 0, got {}", v);
        }
    }
}
