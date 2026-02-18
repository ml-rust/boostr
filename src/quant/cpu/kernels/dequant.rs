//! CPU dequantization kernels
//!
//! Pure-math dequant implementations matching llama.cpp bit-for-bit.
//! These operate on raw byte slices (block data) and write f32 output.

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

/// Dequantize Q4_K blocks to f32
///
/// Q4_K: 256 elements, 144 bytes/block
/// Layout: 2-byte d, 2-byte dmin, 12-byte scales, 128-byte qs
/// Uses 8 sub-blocks of 32 elements with 6-bit scales/mins.
pub fn dequant_q4k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let sc = &block[4..16]; // 12-byte scales array
        let qs = &block[16..144]; // 128-byte quantized values
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Unpack 6-bit scales and mins (matches llama.cpp get_scale_min_k4)
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = sc[i] & 0x3F;
            mins[i] = sc[i + 4] & 0x3F;
        }
        for i in 4..8 {
            scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
            mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
        }

        // 8 sub-blocks of 32 elements
        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;

            let chunk = j / 2;
            let is_high = j % 2 == 1;
            let qs_base = chunk * 32;

            for l in 0..32 {
                let q = if is_high {
                    ((qs[qs_base + l] >> 4) & 0x0F) as f32
                } else {
                    (qs[qs_base + l] & 0x0F) as f32
                };
                out[j * 32 + l] = dl * q - ml;
            }
        }
    }
}

/// Dequantize Q6_K blocks to f32
///
/// Q6_K: 256 elements, 210 bytes/block
/// Layout: 128-byte ql, 64-byte qh, 16-byte scales (i8), 2-byte d
/// Processes in two halves of 128 elements each.
pub fn dequant_q6k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let ql = &block[0..128];
        let qh = &block[128..192];
        // SAFETY: u8 and i8 have identical size and alignment. The slice
        // block[192..208] is valid for the lifetime of `block`, and we create
        // an i8 view of exactly 16 bytes (the Q6_K scale factors are signed).
        let sc: &[i8] = unsafe { std::slice::from_raw_parts(block[192..208].as_ptr().cast(), 16) };
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

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

                out[y_base + l] = d * sc[sc_base + is] as f32 * q1 as f32;
                out[y_base + l + 32] = d * sc[sc_base + is + 2] as f32 * q2 as f32;
                out[y_base + l + 64] = d * sc[sc_base + is + 4] as f32 * q3 as f32;
                out[y_base + l + 96] = d * sc[sc_base + is + 6] as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q4_0_zeros() {
        // All nibbles = 8 (value 0 after subtracting 8), scale = 1.0
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
    fn test_dequant_q4k_zero_scales() {
        // All zeros → all output should be 0 (or just -dmin*min, but with zero scales/mins = 0)
        let block = [0u8; 144];
        let mut output = [0.0f32; 256];
        dequant_q4k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q6k_zero_scales() {
        let block = [0u8; 210];
        let mut output = [0.0f32; 256];
        dequant_q6k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }
}
