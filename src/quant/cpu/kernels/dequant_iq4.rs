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

        // Split-half nibble order, as in llama.cpp's `dequantize_row_iq4_nl`:
        // `y[j]` takes the low nibble, `y[j + QK4_NL/2]` the high nibble of the
        // SAME byte. See the `dequant_simple` module docs.
        for j in 0..16 {
            let byte = qs[j];
            let low = (byte & 0x0F) as usize;
            let high = ((byte >> 4) & 0x0F) as usize;
            out[j] = d * KVALUES_IQ4NL[low] as f32;
            out[j + 16] = d * KVALUES_IQ4NL[high] as f32;
        }
    }
}

/// Dequantize IQ4_XS blocks to f32
///
/// IQ4_XS: 256 elements, 136 bytes/block
///
/// Layout matches llama.cpp's `block_iq4_xs` exactly:
/// `{ ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }`
/// — so `scales_h` is a **2-byte** field at offset 2..4 and `scales_l` occupies
/// 4..8. There is no pad byte.
///
/// 8 sub-blocks of 32 elements, each with a 6-bit scale assembled as
/// `ls = (scales_l nibble) | ((scales_h >> 2*ib) & 3) << 4`, applied as
/// `dl = d * (ls - 32)`. `scales_h` supplies high bits for ALL eight
/// sub-blocks (16 bits = 8 x 2), not just the first four.
///
/// Values use the `KVALUES_IQ4NL` codebook in split-half nibble order within
/// each sub-block (`y[j]` low nibble, `y[j+16]` high nibble of the same byte).
pub fn dequant_iq4_xs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 136;
    const NUM_SUB_BLOCKS: usize = 8;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for sb in 0..NUM_SUB_BLOCKS {
            // 4 low bits from scales_l (one nibble per sub-block), 2 high bits
            // from scales_h (2 bits per sub-block across all 8).
            let sl = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0x0F;
            let sh = ((scales_h >> (2 * sb)) & 0x03) as u8;
            let ls = (sl | (sh << 4)) as i32;
            let sub_scale = d * (ls - 32) as f32;

            let sub_qs = &qs[sb * 16..(sb + 1) * 16];
            let sub_out = &mut out[sb * 32..(sb + 1) * 32];

            for j in 0..16 {
                let byte = sub_qs[j];
                let low = (byte & 0x0F) as usize;
                let high = ((byte >> 4) & 0x0F) as usize;
                sub_out[j] = sub_scale * KVALUES_IQ4NL[low] as f32;
                sub_out[j + 16] = sub_scale * KVALUES_IQ4NL[high] as f32;
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

    /// IQ4_NL uses the same split-half nibble order as Q4_0 (llama.cpp
    /// `dequantize_row_iq4_nl`: `y[j]` low nibble, `y[j + QK4_NL/2]` high
    /// nibble of the SAME byte), with each nibble indexing the non-linear
    /// value table. Built so every element is distinct, so a sequential
    /// `out[2i]`/`out[2i+1]` ordering fails.
    #[test]
    fn iq4_nl_uses_split_half_nibble_order() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        for j in 0..16usize {
            block[2 + j] = (j as u8) | ((15 - j as u8) << 4);
        }

        let mut out = [0.0f32; 32];
        dequant_iq4_nl(&block, &mut out);

        for j in 0..16 {
            assert_eq!(
                out[j], KVALUES_IQ4NL[j] as f32,
                "first half: out[{j}] is the LOW nibble of qs[{j}]"
            );
            assert_eq!(
                out[j + 16],
                KVALUES_IQ4NL[15 - j] as f32,
                "second half: out[{}] is the HIGH nibble of qs[{j}]",
                j + 16
            );
        }
    }

    /// IQ4_XS pins BOTH the sub-block scale assembly and the nibble order.
    ///
    /// `block_iq4_xs` is `{ half d; uint16_t scales_h; uint8_t scales_l[4];
    /// uint8_t qs[128]; }` — `scales_h` is TWO bytes (offset 2..4), `scales_l`
    /// starts at 4, and `scales_h` carries high scale bits for all EIGHT
    /// sub-blocks (2 bits each = 16). Reading `scales_h` as one byte with
    /// `scales_l` at 3 shifts every scale AND drops the high bits of
    /// sub-blocks 4..8.
    #[test]
    fn iq4_xs_scale_assembly_and_nibble_order() {
        let mut block = [0u8; 136];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());

        // Give sub-block 7 the 6-bit scale 33 -> dl = 1.0 * (33 - 32) = 1.0.
        // low nibble 1 (scales_l[3] high nibble), high bits 2 (scales_h bits 14..16).
        let scales_h: u16 = 0b10 << 14;
        block[2..4].copy_from_slice(&scales_h.to_le_bytes());
        block[4 + 3] = 0x10; // scales_l[3] high nibble = 1 -> sub-block 7 low bits

        // Sub-block 7's 16 qs bytes: distinct low/high nibbles.
        for j in 0..16usize {
            block[8 + 7 * 16 + j] = (j as u8) | ((15 - j as u8) << 4);
        }

        let mut out = [0.0f32; 256];
        dequant_iq4_xs(&block, &mut out);

        let sub = &out[7 * 32..8 * 32];
        for j in 0..16 {
            assert_eq!(
                sub[j], KVALUES_IQ4NL[j] as f32,
                "sub-block 7 scale must be (1|2<<4)-32 = 1, and out[{j}] the LOW nibble"
            );
            assert_eq!(
                sub[j + 16],
                KVALUES_IQ4NL[15 - j] as f32,
                "out[{}] must be the HIGH nibble of the same byte",
                j + 16
            );
        }
    }
}
