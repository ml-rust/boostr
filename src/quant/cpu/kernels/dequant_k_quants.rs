//! CPU dequantization kernels for k-quant formats
//!
//! Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K — variable-rate formats with sub-block scales.

use half::f16;

/// Dequantize Q2_K blocks to f32
///
/// Q2_K: 256 elements, 84 bytes/block
/// Layout: scales[16] + qs[64] + d(2) + dmin(2)
/// 16 sub-blocks of 16 elements. scales[i]: low 4 bits = sub-scale, high 4 bits = sub-min.
/// Formula: x = d * sub_scale * q - dmin * sub_min
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

        for (i, out_val) in out.iter_mut().enumerate() {
            let sub_block = i / 16;
            let qs_byte = i / 4;
            let qs_shift = (i % 4) * 2;
            let q = ((qs[qs_byte] >> qs_shift) & 0x03) as f32;
            let dl = d * (sc[sub_block] & 0x0F) as f32;
            let ml = dmin * (sc[sub_block] >> 4) as f32;
            *out_val = dl * q - ml;
        }
    }
}

/// Dequantize Q3_K blocks to f32
///
/// Q3_K: 256 elements, 110 bytes/block
/// Layout: hmask[32] + qs[64] + scales[12] + d(2)
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
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let sc = &block[96..108];
        let d = f16::from_le_bytes([block[108], block[109]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Unpack 16 6-bit scales from 12 bytes (matches llama.cpp)
        let scales = unpack_q3k_scales(sc);

        for (i, out_val) in out.iter_mut().enumerate() {
            let sub_block = i / 16;

            // Low 2 bits from qs (4 values per byte)
            let qs_byte = i / 4;
            let qs_shift = (i % 4) * 2;
            let low2 = (qs[qs_byte] >> qs_shift) & 0x03;

            // High bit from hmask (8 values per byte)
            let hm_byte = i / 8;
            let hm_bit = i % 8;
            let high1 = (hmask[hm_byte] >> hm_bit) & 0x01;

            // Combine to 3-bit unsigned [0, 7], subtract 4 for signed [-4, 3]
            let qu = low2 | (high1 << 2);
            let q = qu as i8 - 4;

            *out_val = d * scales[sub_block] as f32 * q as f32;
        }
    }
}

/// Unpack 16 6-bit scales from 12-byte packed array for Q3_K.
///
/// Matches llama.cpp's scale unpacking using u32 bit manipulation.
/// Returns 16 signed scales (each in [-32, 31]).
fn unpack_q3k_scales(sc: &[u8]) -> [i8; 16] {
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

/// Dequantize Q8_K blocks to f32
///
/// Q8_K: 256 elements, 292 bytes/block
/// Layout: 4-byte f32 d + 256-byte i8 qs + 32-byte i16 bsums
/// Formula: x = q * d (bsums are for dot product optimization only)
/// Note: Q8_K is unique in using f32 scale (not f16).
pub fn dequant_q8k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 292;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        // Q8K uses f32 scale, not f16
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];
        // block[260..292] is bsums[16] (i16), ignored for dequant
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for (out_val, &qs_val) in out.iter_mut().zip(qs.iter()) {
            *out_val = qs_val as i8 as f32 * d;
        }
    }
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
    fn test_dequant_q6k_zero_scales() {
        let block = [0u8; 210];
        let mut output = [0.0f32; 256];
        dequant_q6k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

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
        // Build a Q2_K block with d=1.0, dmin=0.0, sub-scale=1 for all sub-blocks,
        // and all qs bytes = 0b01_01_01_01 (q=1 for every element)
        let mut block = [0u8; 84];
        // scales[0..16]: low nibble = sub_scale=1, high nibble = sub_min=0
        block[0..16].fill(0x01);
        // qs[16..80]: each byte = 0x55 → four 2-bit values of 1
        block[16..80].fill(0x55);
        // d = 1.0
        block[80..82].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // dmin = 0.0
        block[82..84].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());

        let mut output = [0.0f32; 256];
        dequant_q2k(&block, &mut output);

        // x = d * sub_scale * q - dmin * sub_min = 1.0 * 1 * 1 - 0 = 1.0
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 1.0).abs() < 0.01, "elem {i}: expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_q2k_with_min() {
        // d=2.0, dmin=0.5, sub-scale=3, sub-min=2, all q=2
        let mut block = [0u8; 84];
        // scales: low nibble = 3, high nibble = 2 → byte = 0x23
        block[0..16].fill(0x23);
        // qs: all 2-bit values = 2 → 0b10_10_10_10 = 0xAA
        block[16..80].fill(0xAA);
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
        // Build a Q3_K block: d=1.0, hmask=0 (high bit=0), qs=0 (low2=0)
        // q = (0|0) - 4 = -4
        // Scales: we need unpack_q3k_scales to produce known values.
        // With all sc bytes = 0x21 (arbitrary non-zero):
        // aux[0..2] = 0x21212121, aux[2] = 0x21212121
        // Compute expected scales via the helper and verify output.
        let mut block = [0u8; 110];
        // hmask[0..32] = all 0xFF → high bit = 1 for all elements
        block[0..32].fill(0xFF);
        // qs[32..96] = all 0xFF → low2 = 3 for all elements
        block[32..96].fill(0xFF);
        // sc[96..108] = all zeros (scales all = -32 after unpack)
        // d = 0.5
        block[108..110].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());

        let mut output = [0.0f32; 256];
        dequant_q3k(&block, &mut output);

        // qu = 3 | (1 << 2) = 7, q = 7 - 4 = 3
        // scales all = -32 (from all-zero sc bytes)
        // x = 0.5 * (-32) * 3 = -48.0
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-48.0)).abs() < 0.1,
                "elem {i}: expected -48.0, got {v}"
            );
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
    fn test_dequant_q8k_known_values() {
        let mut block = [0u8; 292];
        // f32 scale = 0.5
        block[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        // qs = 10 as i8 → value = 10 * 0.5 = 5.0
        block[4..260].fill(10);

        let mut output = [0.0f32; 256];
        dequant_q8k(&block, &mut output);

        for &v in &output {
            assert!((v - 5.0).abs() < 0.01, "expected 5.0, got {}", v);
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
