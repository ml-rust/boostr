//! Q4_K and Q5_K dequantization kernels

use half::f16;

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
        let (scales, mins) = unpack_q4k_q5k_scales(sc);

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

/// Dequantize Q5_K blocks to f32
///
/// Q5_K: 256 elements, 176 bytes/block
/// Layout: 2-byte d, 2-byte dmin, 12-byte scales, 32-byte qh, 128-byte qs
/// 8 sub-blocks of 32 elements with 6-bit scales/mins (same as Q4_K).
/// 5-bit values: 4 low bits from qs + 1 high bit from qh.
/// Formula: x = d * sc * q - dmin * m
pub fn dequant_q5k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let sc = &block[4..16]; // 12-byte scales array
        let qh = &block[16..48]; // 32-byte high bits
        let qs = &block[48..176]; // 128-byte low nibbles
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Unpack 6-bit scales and mins (same as Q4K)
        let (scales, mins) = unpack_q4k_q5k_scales(sc);

        // 8 sub-blocks of 32 elements
        for j in 0..8 {
            let dl = d * scales[j] as f32;
            let ml = dmin * mins[j] as f32;

            for l in 0..32 {
                let idx = j * 32 + l;

                // Low 4 bits (same packing as Q4K: pairs of sub-blocks share 32 bytes)
                let qs_idx = j * 16 + l / 2;
                let low4 = if l % 2 == 0 {
                    qs[qs_idx] & 0x0F
                } else {
                    (qs[qs_idx] >> 4) & 0x0F
                };

                // High bit from qh (1 bit per element, 8 per byte)
                let qh_byte = idx / 8;
                let qh_bit = idx % 8;
                let high1 = (qh[qh_byte] >> qh_bit) & 0x01;

                // Combine to 5-bit value [0, 31]
                let q = (low4 | (high1 << 4)) as f32;

                out[idx] = dl * q - ml;
            }
        }
    }
}

/// Unpack 6-bit scales and mins from 12-byte packed array for Q4_K and Q5_K.
///
/// Shared between Q4_K and Q5_K (identical scale packing).
pub fn unpack_q4k_q5k_scales(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
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

    (scales, mins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q4k_zero_scales() {
        let block = [0u8; 144];
        let mut output = [0.0f32; 256];
        dequant_q4k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q5k_zero() {
        let block = [0u8; 176];
        let mut output = [0.0f32; 256];
        dequant_q5k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q5k_known_values() {
        // d=1.0, dmin=0.0, all scales=1, all mins=0
        // qh=0 (high bit=0), qs all nibbles = 5 → 5-bit value = 5
        // x = 1.0 * 1 * 5 - 0 = 5.0
        let mut block = [0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&f16::from_f32(0.0).to_le_bytes()); // dmin
        // sc[4..16]: scales[0..4] low 6 bits = 1 → byte = 0x01, mins[4..8] = 0
        block[4..8].fill(0x01); // scales[0..4] = 1
        block[8..12].fill(0x00); // mins[0..4] = 0
        block[12..16].fill(0x00); // scales[4..7] and mins[4..7] = 0 (via upper packing)
        // qh[16..48] = all zeros (high bits = 0)
        // qs[48..176]: nibble 5 for both low and high → 0x55
        block[48..176].fill(0x55);

        let mut output = [0.0f32; 256];
        dequant_q5k(&block, &mut output);

        // Sub-blocks 0-3 have scale=1, sub-blocks 4-7 have scale=0
        // First 128 elements (sub-blocks 0-3): x = 1.0 * 1 * 5 = 5.0
        for (i, &v) in output[..128].iter().enumerate() {
            assert!((v - 5.0).abs() < 0.1, "elem {i}: expected 5.0, got {v}");
        }
        // Last 128 elements (sub-blocks 4-7): scale=0, so x = 0
        for (i, &v) in output[128..].iter().enumerate() {
            assert!(v.abs() < 1e-5, "elem {}: expected ~0, got {}", i + 128, v);
        }
    }

    #[test]
    fn test_unpack_q4k_q5k_scales_basic() {
        // Verify the shared scale unpacking works
        let mut sc = [0u8; 12];
        // First 4 scales: low 6 bits of sc[0..4]
        sc[0] = 10; // scale[0] = 10
        sc[1] = 20; // scale[1] = 20
        let (scales, _mins) = unpack_q4k_q5k_scales(&sc);
        assert_eq!(scales[0], 10);
        assert_eq!(scales[1], 20);
    }
}
