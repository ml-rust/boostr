//! Q2_K and Q3_K dequantization kernels

use half::f16;

/// Dequantize Q2_K blocks to f32
///
/// Q2_K: 256 elements, 84 bytes/block
/// Layout (ggml): scales[16] + qs[64] + d(f16) + dmin(f16)
/// 16 sub-blocks of 16 elements. scales[i]: low 4 bits = sub-scale, high 4 bits = sub-min.
/// Formula: x = d * sub_scale * q - dmin * sub_min
#[allow(clippy::needless_range_loop)]
pub fn dequant_q2k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 84;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let sc = &block[0..16];
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Matches llama.cpp: process in 2 groups of 128, each with 4 shift levels
        let mut y = 0;
        let mut is = 0;
        let mut q_offset = 0;
        for _n in 0..2 {
            // 2 groups of 128 elements, each using 32 bytes of qs
            let q = &qs[q_offset..];
            for shift in (0..8).step_by(2) {
                // Sub-block A: 16 elements from q[0..16]
                let dl = d * (sc[is] & 0x0F) as f32;
                let ml = dmin * (sc[is] >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    out[y] = dl * ((q[l] >> shift) & 3) as f32 - ml;
                    y += 1;
                }
                // Sub-block B: 16 elements from q[16..32]
                let dl = d * (sc[is] & 0x0F) as f32;
                let ml = dmin * (sc[is] >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    out[y] = dl * ((q[16 + l] >> shift) & 3) as f32 - ml;
                    y += 1;
                }
            }
            q_offset += 32;
        }
    }
}

/// Dequantize Q3_K blocks to f32
///
/// Q3_K: 256 elements, 110 bytes/block
/// Layout (ggml): hmask[32] + qs[64] + scales[12] + d(f16)
/// 16 sub-blocks of 16 elements with 6-bit signed sub-block scales.
/// 3-bit values: 2 low bits from qs + 1 high bit from hmask.
/// Formula: x = d * sub_scale * (q - 4)
pub fn dequant_q3k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 110;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let hm = &block[0..32];
        let qs = &block[32..96];
        let sc = &block[96..108];
        let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Unpack 16 6-bit scales from 12 bytes (matches llama.cpp)
        let scales = unpack_q3k_scales(sc);

        // Matches llama.cpp: 2 groups of 128, each with 4 shift levels, running hmask bit
        let mut y = 0;
        let mut is = 0;
        let mut m: u8 = 1; // running hmask bit
        let mut q_offset = 0;
        for _n in 0..2 {
            let q = &qs[q_offset..];
            for shift in (0..8).step_by(2) {
                let dl = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let low2 = (q[l] >> shift) & 3;
                    let high_sub = if hm[l] & m != 0 { 0i8 } else { 4i8 };
                    out[y] = dl * (low2 as i8 - high_sub) as f32;
                    y += 1;
                }
                let dl = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let low2 = (q[16 + l] >> shift) & 3;
                    let high_sub = if hm[16 + l] & m != 0 { 0i8 } else { 4i8 };
                    out[y] = dl * (low2 as i8 - high_sub) as f32;
                    y += 1;
                }
                m = m.wrapping_shl(1);
            }
            q_offset += 32;
        }
    }
}

/// Unpack 16 6-bit scales from 12-byte packed array for Q3_K.
///
/// Matches llama.cpp's scale unpacking using u32 bit manipulation.
/// Returns 16 signed scales (each in [-32, 31]).
pub fn unpack_q3k_scales(sc: &[u8]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
    aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
    aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);

    let tmp = aux[2];
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let a0 = aux[0];
    let a1 = aux[1];
    aux[0] = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    aux[2] = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&aux[0].to_le_bytes());
    bytes[4..8].copy_from_slice(&aux[1].to_le_bytes());
    bytes[8..12].copy_from_slice(&aux[2].to_le_bytes());
    bytes[12..16].copy_from_slice(&aux[3].to_le_bytes());

    let mut scales = [0i8; 16];
    for (scale, &byte) in scales.iter_mut().zip(bytes.iter()) {
        *scale = byte as i8 - 32;
    }
    scales
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q2k_zero() {
        let block = [0u8; 84];
        let mut output = [0.0f32; 256];
        dequant_q2k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q2k_known_values() {
        // Build a Q2_K block with d=1.0, dmin=0.0, all scales=1, all qs=0x55
        // Layout: scales[0..16] + qs[16..80] + d[80..82] + dmin[82..84]
        //
        // With the shift-based packing, qs byte 0x55 = 0b01_01_01_01:
        // At shift 0: (0x55 >> 0) & 3 = 1
        // At shift 2: (0x55 >> 2) & 3 = 1
        // At shift 4: (0x55 >> 4) & 3 = 1
        // At shift 6: (0x55 >> 6) & 3 = 1
        // So all q values = 1 regardless of shift.
        let mut block = [0u8; 84];
        block[0..16].fill(0x01); // scales: sub_scale=1, sub_min=0
        block[16..80].fill(0x55); // qs: q=1 at all shifts
        block[80..82].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[82..84].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());

        let mut output = [0.0f32; 256];
        dequant_q2k(&block, &mut output);

        // x = 1.0 * 1 * 1 - 0.0 * 0 = 1.0
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 1.0).abs() < 0.01, "elem {i}: expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q2k_with_min() {
        // d=2.0, dmin=0.5, sub-scale=3, sub-min=2
        // qs = 0xAA = 0b10_10_10_10 → q=2 at all shifts
        let mut block = [0u8; 84];
        block[0..16].fill(0x23); // scales: low=3, high=2
        block[16..80].fill(0xAA); // qs: q=2 at all shifts
        block[80..82].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[82..84].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        let mut output = [0.0f32; 256];
        dequant_q2k(&block, &mut output);

        // x = 2.0 * 3 * 2 - 0.5 * 2 = 12.0 - 1.0 = 11.0
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 11.0).abs() < 0.1, "elem {i}: expected 11.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q3k_zero() {
        let block = [0u8; 110];
        let mut output = [0.0f32; 256];
        dequant_q3k(&block, &mut output);

        // d = 0.0, so output = 0 regardless of scale/q
        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q3k_known_values() {
        // Build a Q3_K block: d=0.5, hmask=0xFF, qs=0xFF
        // Layout: hmask[0..32] + qs[32..96] + scales[96..108] + d[108..110]
        //
        // With shift-based packing and running mask m:
        //   qs=0xFF → (0xFF >> shift) & 3 = 3 at all shifts
        //   hmask=0xFF → (hm[l] & m) is always nonzero → subtract 0
        //   so q = 3 - 0 = 3
        //   scales all = -32 (from all-zero sc bytes)
        //   x = 0.5 * (-32) * 3 = -48.0
        let mut block = [0u8; 110];
        block[0..32].fill(0xFF); // hmask: all bits set
        block[32..96].fill(0xFF); // qs: all bits set
        // sc[96..108] = all zeros
        block[108..110].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        let mut output = [0.0f32; 256];
        dequant_q3k(&block, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-48.0)).abs() < 0.1,
                "elem {i}: expected -48.0, got {v}"
            );
        }
    }

    #[test]
    fn test_unpack_q3k_scales_zeros() {
        // All zeros → each byte is 0, subtract 32 → all -32
        let sc = [0u8; 12];
        let scales = unpack_q3k_scales(&sc);
        for &s in &scales {
            assert_eq!(s, -32);
        }
    }

    #[test]
    fn test_unpack_q3k_scales_all_ones() {
        // All 0xFF bytes
        let sc = [0xFF; 12];
        let scales = unpack_q3k_scales(&sc);
        // With all-FF input, each unpacked 6-bit value is 0x3F = 63
        // 63 - 32 = 31 (max positive scale)
        for &s in &scales {
            assert_eq!(s, 31);
        }
    }

    #[test]
    #[ignore] // temp debugging test
    fn test_dequant_q3k_real_block() {
        // Real block from blk.0.attn_v.weight block 10 of mistral Q2_K model
        let block: [u8; 110] = [
            249, 99, 245, 234, 226, 236, 45, 116, 159, 178, 189, 173, 255, 243, 26, 125, 222, 253,
            238, 81, 247, 255, 191, 230, 74, 99, 179, 247, 70, 110, 203, 143, 238, 36, 144, 36,
            114, 66, 196, 206, 0, 61, 12, 12, 18, 224, 193, 9, 19, 0, 92, 104, 251, 17, 16, 138,
            255, 193, 98, 128, 3, 103, 20, 0, 0, 143, 16, 6, 3, 79, 4, 200, 50, 244, 48, 99, 20,
            24, 248, 204, 86, 1, 2, 12, 0, 0, 68, 74, 131, 147, 55, 33, 158, 192, 79, 63, 81, 211,
            209, 159, 0, 207, 132, 227, 55, 195, 39, 36, 44, 6,
        ];
        // Expected from Python (llama.cpp algorithm): first 8 values
        let expected_first8: [f32; 8] = [0.0032, 0.0, 0.0, -0.0064, -0.0032, -0.0032, 0.0, -0.0032];

        let mut output = [0.0f32; 256];
        dequant_q3k(&block, &mut output);

        for (i, &exp) in expected_first8.iter().enumerate() {
            assert!(
                (output[i] - exp).abs() < 0.001,
                "elem {i}: expected {exp}, got {}",
                output[i]
            );
        }
    }
}
