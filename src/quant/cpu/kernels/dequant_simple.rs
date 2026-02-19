//! CPU dequantization kernels for simple block formats
//!
//! Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 — fixed-size blocks with straightforward layouts.

use half::f16;

/// Dequantize Q4_0 blocks to f32
///
/// Q4_0: 32 elements, 18 bytes/block (2-byte f16 scale + 16 bytes of nibbles)
/// Formula: x = (nibble - 8) * scale
pub fn dequant_q4_0(blocks: &[u8], output: &mut [f32]) {
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
            let low = (byte & 0x0F) as i8 - 8;
            let high = ((byte >> 4) & 0x0F) as i8 - 8;
            out[i * 2] = low as f32 * d;
            out[i * 2 + 1] = high as f32 * d;
        }
    }
}

/// Dequantize Q4_1 blocks to f32
///
/// Q4_1: 32 elements, 20 bytes/block (2-byte f16 scale + 2-byte f16 min + 16 bytes of nibbles)
/// Formula: x = d * nibble + m
pub fn dequant_q4_1(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qs = &block[4..20];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = qs[i];
            let low = (byte & 0x0F) as f32;
            let high = ((byte >> 4) & 0x0F) as f32;
            out[i * 2] = d * low + m;
            out[i * 2 + 1] = d * high + m;
        }
    }
}

/// Dequantize Q5_0 blocks to f32
///
/// Q5_0: 32 elements, 22 bytes/block (2-byte f16 scale + 4-byte high bits + 16 bytes low nibbles)
/// Formula: x = (5-bit value - 16) * scale
pub fn dequant_q5_0(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 22;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = qs[i];
            let low_nibble = byte & 0x0F;
            let high_nibble = (byte >> 4) & 0x0F;

            let hbit_low = ((qh_bits >> (i * 2)) & 1) as u8;
            let hbit_high = ((qh_bits >> (i * 2 + 1)) & 1) as u8;

            let val_low = ((hbit_low << 4) | low_nibble) as i8 - 16;
            let val_high = ((hbit_high << 4) | high_nibble) as i8 - 16;

            out[i * 2] = val_low as f32 * d;
            out[i * 2 + 1] = val_high as f32 * d;
        }
    }
}

/// Dequantize Q5_1 blocks to f32
///
/// Q5_1: 32 elements, 24 bytes/block (2-byte f16 d + 2-byte f16 m + 4-byte high bits + 16 bytes low nibbles)
/// Formula: x = d * 5-bit value + m
pub fn dequant_q5_1(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 24;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh_bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..16 {
            let byte = qs[i];
            let low_nibble = byte & 0x0F;
            let high_nibble = (byte >> 4) & 0x0F;

            let hbit_low = ((qh_bits >> (i * 2)) & 1) as u8;
            let hbit_high = ((qh_bits >> (i * 2 + 1)) & 1) as u8;

            let val_low = (low_nibble | (hbit_low << 4)) as f32;
            let val_high = (high_nibble | (hbit_high << 4)) as f32;

            out[i * 2] = d * val_low + m;
            out[i * 2 + 1] = d * val_high + m;
        }
    }
}

/// Dequantize Q8_0 blocks to f32
///
/// Q8_0: 32 elements, 34 bytes/block (2-byte f16 scale + 32 bytes of i8 values)
/// Formula: x = qs\[i\] * scale
pub fn dequant_q8_0(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..32 {
            out[i] = qs[i] as i8 as f32 * d;
        }
    }
}

/// Dequantize Q8_1 blocks to f32
///
/// Q8_1: 32 elements, 36 bytes/block (2-byte f16 d + 2-byte f16 s (sum, ignored) + 32 bytes i8)
/// Formula: x = q * d (the s field is a precomputed sum for dot products, not used in dequant)
pub fn dequant_q8_1(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 36;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        // block[2..4] is `s` (sum), ignored for dequant
        let qs = &block[4..36];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for i in 0..32 {
            out[i] = qs[i] as i8 as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q4_0_zeros() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..18].fill(0x88); // nibbles: low=8, high=8 → 0 after -8

        let mut output = [0.0f32; 32];
        dequant_q4_0(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected 0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q4_0_known_values() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        // nibble 9 → 9-8 = 1, so value = 1 * 2.0 = 2.0
        block[2..18].fill(0x99);

        let mut output = [0.0f32; 32];
        dequant_q4_0(&block, &mut output);

        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q4_1_known_values() {
        let mut block = [0u8; 20];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&f16::from_f32(1.0).to_le_bytes()); // m
        // nibble = 3 → value = 2.0 * 3 + 1.0 = 7.0
        block[4..20].fill(0x33);

        let mut output = [0.0f32; 32];
        dequant_q4_1(&block, &mut output);

        for &v in &output {
            assert!((v - 7.0).abs() < 0.01, "expected 7.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q5_0_known_values() {
        // scale=1.0, all high bits=0, all low nibbles=0 → 5-bit value=0, result=(0-16)*1.0=-16
        let mut block = [0u8; 22];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());

        let mut output = [0.0f32; 32];
        dequant_q5_0(&block, &mut output);

        for &v in &output {
            assert!((v - (-16.0)).abs() < 0.01, "expected -16.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q5_0_midpoint() {
        // 5-bit value = 16 → (16-16)*scale = 0
        let mut block = [0u8; 22];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // qh = all 1s (high bit = 1), low nibble = 0 → 5-bit value = 16
        block[2..6].fill(0xFF);

        let mut output = [0.0f32; 32];
        dequant_q5_0(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 0.01, "expected 0.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q5_1_known_values() {
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes()); // d
        block[2..4].copy_from_slice(&f16::from_f32(1.0).to_le_bytes()); // m
        // qh = all zeros, low nibble = 2 → 5-bit = 2, value = 0.5*2 + 1.0 = 2.0
        block[8..24].fill(0x22);

        let mut output = [0.0f32; 32];
        dequant_q5_1(&block, &mut output);

        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q8_0_known_values() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        // All qs = 4 (as i8), so value = 4 * 0.5 = 2.0
        block[2..34].fill(4);

        let mut output = [0.0f32; 32];
        dequant_q8_0(&block, &mut output);

        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q8_1_known_values() {
        let mut block = [0u8; 36];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes()); // d
        // block[2..4] is s (sum), don't care
        // qs = 6 as i8 → value = 6 * 0.5 = 3.0
        block[4..36].fill(6);

        let mut output = [0.0f32; 32];
        dequant_q8_1(&block, &mut output);

        for &v in &output {
            assert!((v - 3.0).abs() < 0.01, "expected 3.0, got {}", v);
        }
    }
}
