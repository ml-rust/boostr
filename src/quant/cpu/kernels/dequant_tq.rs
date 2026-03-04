//! CPU dequantization kernels for TQ formats
//!
//! TQ2_0, TQ1_0
use half::f16;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
