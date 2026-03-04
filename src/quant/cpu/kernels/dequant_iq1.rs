//! CPU dequantization kernels for IQ1 formats
//!
//! IQ1_S, IQ1_M
use half::f16;

/// Dequantize IQ1_S blocks to f32
///
/// IQ1_S: 256 elements, 50 bytes/block
/// Very low bit-rate (1.5625 bpw), uses ternary grid-based encoding
/// Layout: f16 d (2B) + qs (32B) + qh (16B)
#[allow(clippy::needless_range_loop)]
pub fn dequant_iq1_s(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 50;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34]; // 32 bytes quant data
        let qh = &block[34..50]; // 16 bytes high bits

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 16 sub-blocks of 16 values
        for sb in 0..16 {
            let qs_val = u16::from_le_bytes([qs[sb * 2], qs[sb * 2 + 1]]);
            let grid_idx = (qs_val & 0x0FFF) as usize;
            let sign_bits = qh[sb];

            let sub_out = &mut out[sb * 16..(sb + 1) * 16];

            // IQ1_S grid values are ternary {-1, 0, 1}
            // Grid index encodes 16 values in base-3 style
            let mut grid_val = grid_idx as u32;
            for k in 0..16 {
                let t = (grid_val % 3) as i8 - 1; // {0,1,2} -> {-1,0,1}
                let sign = if (sign_bits >> (k % 8)) & 1 != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };
                sub_out[k] = d * t as f32 * sign;
                grid_val /= 3;
            }
        }
    }
}

/// Dequantize IQ1_M blocks to f32
///
/// IQ1_M: 256 elements, 56 bytes/block
/// No simple f16 d; uses 3-bit packed scales
/// Layout: 6B scale data + f16 d (2B) + qs (32B) + qh (16B)
#[allow(clippy::needless_range_loop)]
pub fn dequant_iq1_m(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 56;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        // IQ1_M layout: scales(6B) + d(2B) + qs(32B) + qh(16B) = 56B
        // But d is at the END in the ggml structure, let me recheck...
        // Actually from llama.cpp: d is at offset 0-1 (first 2 bytes)
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_data = &block[2..8]; // 6 bytes of 3-bit packed scales
        let qs = &block[8..40]; // 32 bytes quant data
        let qh = &block[40..56]; // 16 bytes high bits

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 16 sub-blocks of 16 values
        for sb in 0..16 {
            // Extract 3-bit scale for this sub-block
            let scale_bit_offset = sb * 3;
            let byte_idx = scale_bit_offset / 8;
            let bit_offset = scale_bit_offset % 8;
            let raw = if byte_idx + 1 < 6 {
                let lo = scales_data[byte_idx] as u16;
                let hi = scales_data[byte_idx + 1] as u16;
                ((lo | (hi << 8)) >> bit_offset) & 0x07
            } else if byte_idx < 6 {
                ((scales_data[byte_idx] >> bit_offset) & 0x07) as u16
            } else {
                0
            };
            let sub_scale = d * (raw as f32 + 0.5);

            let qs_val = u16::from_le_bytes([qs[sb * 2], qs[sb * 2 + 1]]);
            let grid_idx = (qs_val & 0x0FFF) as usize;
            let sign_bits = qh[sb];

            let sub_out = &mut out[sb * 16..(sb + 1) * 16];

            let mut grid_val = grid_idx as u32;
            for k in 0..16 {
                let t = (grid_val % 3) as i8 - 1;
                let sign = if (sign_bits >> (k % 8)) & 1 != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };
                sub_out[k] = sub_scale * t as f32 * sign;
                grid_val /= 3;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_iq1_s_zeros() {
        let block = [0u8; 50];
        let mut output = [0.0f32; 256];
        dequant_iq1_s(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequant_iq1_m_zeros() {
        let block = [0u8; 56];
        let mut output = [0.0f32; 256];
        dequant_iq1_m(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }
}
