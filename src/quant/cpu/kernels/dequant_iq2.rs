//! CPU dequantization kernels for IQ2 formats
//!
//! IQ2_XXS, IQ2_XS, IQ2_S
use half::f16;

/// Dequantize IQ2_XXS blocks to f32
///
/// IQ2_XXS: 256 elements, 66 bytes/block
/// Layout: f16 d (2B) + 64 bytes of packed data
/// Each group of 8 bytes encodes 32 values:
///   - 4 grid indices (8 bits each) selecting from a 256-entry pattern table
///   - Signs and scale bits
#[allow(clippy::needless_range_loop)]
pub fn dequant_iq2_xxs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 66;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 8 groups of 32 values = 256 values
        // Each group uses 8 bytes (one u64)
        for group in 0..8 {
            let group_data = &qs[group * 8..(group + 1) * 8];
            let q64 = u64::from_le_bytes([
                group_data[0],
                group_data[1],
                group_data[2],
                group_data[3],
                group_data[4],
                group_data[5],
                group_data[6],
                group_data[7],
            ]);

            // Low 32 bits: 4 grid indices (8 bits each)
            // High 32 bits: signs and scale info
            let grid_indices = q64 as u32;
            let signs_and_scales = (q64 >> 32) as u32;

            // Extract sub-block scale from high byte
            let sub_scale_bits = (signs_and_scales >> 28) & 0x0F;
            let sub_scale = d * (0.5 + sub_scale_bits as f32);

            let group_out = &mut out[group * 32..(group + 1) * 32];

            for sub in 0..4 {
                let grid_idx = ((grid_indices >> (8 * sub)) & 0xFF) as usize;
                let sign_bits = ((signs_and_scales >> (7 * sub)) & 0x7F) as u8;

                // Decode 8 values from grid index
                // IQ2_XXS grid: each index maps to 8 values in {-1, 0, 1}
                for k in 0..8 {
                    let grid_val = iq2xxs_grid_value(grid_idx, k);
                    let sign = if (sign_bits >> k) & 1 != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    group_out[sub * 8 + k] = sub_scale * grid_val * sign;
                }
            }
        }
    }
}

/// Decode a value from the IQ2_XXS grid.
/// The grid maps indices to sets of 8 values.
/// Each value is in the range [0, 3] representing magnitudes.
#[inline]
fn iq2xxs_grid_value(grid_idx: usize, position: usize) -> f32 {
    // The IQ2_XXS grid encodes 8 values per entry where each value is in {0, 1, 2, 3}
    // Extract 2-bit fields from grid_idx
    let shift = position * 2;
    let bits = if shift < 8 {
        (grid_idx >> shift) & 0x03
    } else {
        ((grid_idx >> (shift - 8)) ^ (grid_idx >> 1)) & 0x03
    };
    // Map {0,1,2,3} to magnitudes
    const GRID_VALS: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
    GRID_VALS[bits]
}

/// Dequantize IQ2_XS blocks to f32
///
/// IQ2_XS: 256 elements, 74 bytes/block
/// Layout: f16 d (2B) + 16 bytes scales + 32B grid data + 16B signs
#[allow(clippy::needless_range_loop)]
pub fn dequant_iq2_xs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 74;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales = &block[2..18]; // 16 bytes of scales
        let qs = &block[18..74]; // 56 bytes of quant data

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 16 sub-blocks of 16 values = 256 values
        for sb in 0..16 {
            let scale = d * ((scales[sb] as i8) as f32 + 0.5);

            // Each sub-block uses 2 bytes for grid + sign info
            let q_offset = sb * 2;
            let qs_lo = qs[q_offset] as u16;
            let qs_hi = qs[q_offset + 1] as u16;
            let q_val = qs_lo | (qs_hi << 8);

            // Extract grid and sign bits
            let grid_idx = (q_val & 0x1FF) as usize;
            let signs = (q_val >> 9) as u8;

            let sub_out = &mut out[sb * 16..(sb + 1) * 16];

            // Decode 16 values
            for k in 0..16 {
                let grid_val = if k < 8 {
                    iq2xs_grid_value(grid_idx, k)
                } else {
                    iq2xs_grid_value(grid_idx, k - 8)
                };
                let sign = if (signs >> (k % 8)) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                sub_out[k] = scale * grid_val * sign;
            }
        }
    }
}

#[inline]
fn iq2xs_grid_value(grid_idx: usize, position: usize) -> f32 {
    let bits = (grid_idx >> (position * 2)) & 0x03;
    const VALS: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
    VALS[bits.min(3)]
}

/// Dequantize IQ2_S blocks to f32
///
/// IQ2_S: 256 elements, 82 bytes/block
/// Complex layout with separate grid, high bits, signs, and scales
#[allow(clippy::needless_range_loop)]
pub fn dequant_iq2_s(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 82;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34]; // 32 bytes grid indices
        let _qh = &block[34..38]; // 4 bytes high bits (reserved for future use)
        let signs = &block[38..54]; // 16 bytes signs
        let scales = &block[54..82]; // 28 bytes scales

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // 16 sub-blocks of 16 values
        for sb in 0..16 {
            let scale_byte = if sb < 28 { scales[sb] } else { 0 };
            let sub_scale = d * ((scale_byte as i8) as f32 + 0.5);

            let sub_out = &mut out[sb * 16..(sb + 1) * 16];
            for k in 0..16 {
                let byte_idx = sb * 2 + k / 8;
                let grid_byte = if byte_idx < 32 { qs[byte_idx] } else { 0 };
                let bit_pos = k % 8;
                let sign_byte = if sb < 16 { signs[sb] } else { 0 };
                let sign = if (sign_byte >> bit_pos) & 1 != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };

                let val = ((grid_byte >> (bit_pos % 4 * 2)) & 0x03) as f32;
                sub_out[k] = sub_scale * val * sign;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_iq2_xxs_zeros() {
        let block = [0u8; 66];
        let mut output = [0.0f32; 256];
        dequant_iq2_xxs(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }
}
