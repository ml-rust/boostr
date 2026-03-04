//! CPU dequantization kernels for IQ4 formats
//!
//! IQ4_NL, IQ4_XS
use crate::quant::tables::KVALUES_IQ4NL;
use half::f16;

/// Dequantize IQ4_NL blocks to f32
///
/// IQ4_NL: 32 elements, 18 bytes/block (f16 scale + 16 bytes nibbles)
/// Uses non-linear codebook: x = scale * KVALUES_IQ4NL[nibble]
pub fn dequant_iq4_nl(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = qs[i];
            let low = (byte & 0x0F) as usize;
            let high = ((byte >> 4) & 0x0F) as usize;
            out[i * 2] = d * KVALUES_IQ4NL[low] as f32;
            out[i * 2 + 1] = d * KVALUES_IQ4NL[high] as f32;
        }
    }
}

/// Dequantize IQ4_XS blocks to f32
///
/// IQ4_XS: 256 elements, 136 bytes/block
/// Layout: f16 d (2B) + scales_h (1B) + scales_l (4B) + pad (1B) + qs (128B)
/// 8 sub-blocks of 32 elements each, each with a 6-bit scale
/// Uses KVALUES_IQ4NL codebook for value lookup
pub fn dequant_iq4_xs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 136;
    const NUM_SUB_BLOCKS: usize = 8;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = block[2];
        let scales_l = &block[3..7];
        let _scales_extra = block[7]; // Reserved byte for potential extra scale bits
        // qs starts at offset 8 (after 2B d + 1B scales_h + 4B scales_l + 1B pad)
        let qs = &block[8..136];

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for sb in 0..NUM_SUB_BLOCKS {
            // 6-bit scale: 4 low bits from scales_l, 2 high bits from scales_h
            // scales_l stores 4 bits per sub-block (8 sub-blocks * 4 bits = 32 bits = 4 bytes)
            let sl = if sb % 2 == 0 {
                scales_l[sb / 2] & 0x0F
            } else {
                (scales_l[sb / 2] >> 4) & 0x0F
            };
            // scales_h stores 2 bits per sub-block for the first 4 sub-blocks (4*2=8 bits=1 byte)
            // For sub-blocks 4-7, we only use the low 4 bits from scales_l (no high bits)
            let sh = if sb < 4 {
                (scales_h >> (2 * sb)) & 0x03
            } else {
                0
            };
            let scale_6bit = sl | (sh << 4);
            let sub_scale = d * (scale_6bit as i32 - 32) as f32;

            let sub_qs = &qs[sb * 16..(sb + 1) * 16];
            let sub_out = &mut out[sb * 32..(sb + 1) * 32];

            for i in 0..16 {
                let byte = sub_qs[i];
                let low = (byte & 0x0F) as usize;
                let high = ((byte >> 4) & 0x0F) as usize;
                sub_out[i * 2] = sub_scale * KVALUES_IQ4NL[low] as f32;
                sub_out[i * 2 + 1] = sub_scale * KVALUES_IQ4NL[high] as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_iq4_nl_zeros() {
        let block = [0u8; 18];
        let mut output = [0.0f32; 32];
        dequant_iq4_nl(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequant_iq4_nl_known_values() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // All nibbles = 8 (index 8) → KVALUES_IQ4NL[8] = 1
        block[2..18].fill(0x88);
        let mut output = [0.0f32; 32];
        dequant_iq4_nl(&block, &mut output);
        for &v in &output {
            assert!((v - 1.0).abs() < 0.01, "expected 1.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_iq4_xs_zeros() {
        let block = [0u8; 136];
        let mut output = [0.0f32; 256];
        dequant_iq4_xs(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }
}
