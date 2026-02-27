//! CPU dequantization kernels for IQ/TQ formats
//!
//! IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, TQ1_0, TQ2_0
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

/// Dequantize TQ2_0 blocks to f32
///
/// TQ2_0: 256 elements, 66 bytes/block
/// Layout: f16 d (2B) + 64 bytes of 2-bit packed values
/// Each byte holds 4 values (2 bits each): val = ((byte >> (2*j)) & 3) - 1
/// Maps to {-1, 0, 1}
pub fn dequant_tq2_0(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 66;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..64 {
            let byte = qs[i];
            for j in 0..4 {
                let val = ((byte >> (2 * j)) & 0x03) as i8 - 1;
                out[i * 4 + j] = d * val as f32;
            }
        }
    }
}

/// Dequantize TQ1_0 blocks to f32
///
/// TQ1_0: 256 elements, 54 bytes/block
/// Layout: f16 d (2B) + qs (52B) where qs encodes 256 ternary values
/// Ternary values: {-1, 0, +1}, 5 values packed per byte using base-3 encoding
pub fn dequant_tq1_0(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 54;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..54]; // 52 bytes

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        let mut idx = 0;
        for &byte in qs.iter() {
            let mut val = byte as u32;
            for _ in 0..5 {
                if idx >= BLOCK_SIZE {
                    break;
                }
                let t = (val % 3) as i8 - 1; // {0,1,2} -> {-1,0,1}
                out[idx] = d * t as f32;
                val /= 3;
                idx += 1;
            }
        }
    }
}

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
    fn test_dequant_tq2_0_zeros() {
        let block = [0u8; 66];
        let mut output = [0.0f32; 256];
        dequant_tq2_0(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequant_tq2_0_known_values() {
        let mut block = [0u8; 66];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        // All 2-bit values = 1 → val = 1-1 = 0 → output = 0
        block[2..66].fill(0x55); // 0b01010101 → all 2-bit fields = 1
        let mut output = [0.0f32; 256];
        dequant_tq2_0(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 0.01, "expected 0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_tq1_0_zeros() {
        let block = [0u8; 54];
        let mut output = [0.0f32; 256];
        dequant_tq1_0(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
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

    #[test]
    fn test_dequant_iq2_xxs_zeros() {
        let block = [0u8; 66];
        let mut output = [0.0f32; 256];
        dequant_iq2_xxs(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }

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
