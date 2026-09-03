//! CPU dequantization kernels for TQ formats
//!
//! TQ2_0, TQ1_0
//!
//! Both ternary formats store the f16 scale `d` at the END of the block, not
//! the start, and both order elements level-major rather than byte-major:
//!
//! ```text
//!   TQ1_0 (54B): qs[0..48], qh[48..52], d[52..54]
//!   TQ2_0 (66B): qs[0..64],             d[64..66]
//! ```
//!
//! Reading `d` from offset 0 yields a scale assembled from packed trits — a
//! tiny value that leaves the tensor finite and plausibly shaped while every
//! weight in it is wrong. The CUDA side keeps the same layout in
//! `cuda/kernels/decode.cuh`; the two must stay in agreement, and
//! `tests/gguf_conformance_llama_cpp.rs` gates both against llama.cpp.
use half::f16;

/// Ternary value {-1, 0, 1} of element `elem` of a TQ1_0 block.
///
/// TQ1_0 packs FIVE trits per byte in base 3 and does not decode them by
/// repeated division. llama.cpp stores each byte pre-scaled so a trit is
/// recovered by a WRAPPING 8-bit multiply against a power of three followed by
/// a multiply-shift. The wrap is load-bearing: widening it changes the result.
///
/// The 256 elements come from three differently shaped runs, in this order:
/// `[0, 160)` is `qs[0..32]` over 5 levels, `[160, 240)` is `qs[32..48]` over
/// 5 levels, and `[240, 256)` is `qh[0..4]` over 4 levels.
#[inline]
fn tq1_0_trit(block: &[u8], elem: usize) -> i32 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let (byte, level) = if elem < 160 {
        (block[elem % 32], elem / 32)
    } else if elem < 240 {
        let r = elem - 160;
        (block[32 + r % 16], r / 16)
    } else {
        let r = elem - 240;
        (block[48 + r % 4], r / 4)
    };
    let q = byte.wrapping_mul(POW3[level]);
    ((u16::from(q) * 3) >> 8) as i32 - 1
}

/// Ternary value {-1, 0, 1} of element `elem` of a TQ2_0 block.
#[inline]
fn tq2_0_trit(block: &[u8], elem: usize) -> i32 {
    let group = elem / 128; // 128 elements per 32-byte group
    let r = elem % 128;
    let level = r / 32; // which 2-bit field
    let m = r % 32; // which byte in the group
    i32::from((block[group * 32 + m] >> (2 * level)) & 0x03) - 1
}

/// Dequantize TQ2_0 blocks to f32
///
/// TQ2_0: 256 elements, 66 bytes/block. Layout `qs[64] + d:f16`.
pub fn dequant_tq2_0(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 66;
    const D_OFFSET: usize = 64;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];
        for (elem, slot) in out.iter_mut().enumerate() {
            *slot = d * tq2_0_trit(block, elem) as f32;
        }
    }
}

/// Dequantize TQ1_0 blocks to f32
///
/// TQ1_0: 256 elements, 54 bytes/block. Layout `qs[48] + qh[4] + d:f16`.
pub fn dequant_tq1_0(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 54;
    const D_OFFSET: usize = 52;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];
        for (elem, slot) in out.iter_mut().enumerate() {
            *slot = d * tq1_0_trit(block, elem) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A byte of 49 is the base-3 packing of digits `[0, 1, 2, 0, 1]`, i.e.
    /// trits `[-1, 0, 1, -1, 0]` — llama.cpp stores `(46 * 256 + 242) / 243`
    /// where 46 is `0*81 + 1*27 + 2*9 + 0*3 + 1`. Levels 2, 3 and 4 all
    /// overflow 8 bits during decode, so this one value also proves the
    /// multiply wraps rather than widens.
    const TQ1_PACKED: u8 = 49;
    const TQ1_TRITS: [f32; 5] = [-1.0, 0.0, 1.0, -1.0, 0.0];

    /// `0b00_10_01_00` — 2-bit fields 0..4 are 0, 1, 2, 0, i.e. trits
    /// -1, 0, 1, -1 at four positions 32 elements apart.
    const TQ2_PACKED: u8 = 0x24;
    const TQ2_TRITS: [f32; 4] = [-1.0, 0.0, 1.0, -1.0];

    #[test]
    fn test_dequant_tq2_0_zeros() {
        let block = [0u8; 66];
        let mut output = [0.0f32; 256];
        dequant_tq2_0(&block, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }

    /// Pins the scale to the END of the block and the element order to
    /// level-major. Reading `d` from offset 0, or emitting the four 2-bit
    /// fields of a byte to adjacent positions, both fail here.
    #[test]
    fn test_dequant_tq2_0_layout() {
        let mut block = [0u8; 66];
        block[64..66].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[0] = TQ2_PACKED; // group 0, byte 0
        block[32] = TQ2_PACKED; // group 1, byte 0
        let mut output = [0.0f32; 256];
        dequant_tq2_0(&block, &mut output);

        for (level, &trit) in TQ2_TRITS.iter().enumerate() {
            for (group, base) in [(0usize, 0usize), (1, 128)] {
                let idx = base + level * 32;
                assert_eq!(
                    output[idx],
                    2.0 * trit,
                    "group {group} level {level} (element {idx})"
                );
            }
        }
        // Every element fed by a zero byte decodes to trit -1, never to 0:
        // the packed field is 0 and TQ2_0 maps 0 -> -1.
        assert_eq!(output[1], -2.0);
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

    /// Pins all three of TQ1_0's differently shaped runs at once, plus the
    /// scale offset. A byte-major "divide by 3 repeatedly" decode, a `d` read
    /// from offset 0, or a decode that ignores the `qh` tail all fail here.
    #[test]
    fn test_dequant_tq1_0_layout() {
        let mut block = [0u8; 54];
        block[52..54].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[0] = TQ1_PACKED; // qs0 run: elements 0, 32, 64, 96, 128
        block[32] = TQ1_PACKED; // qs1 run: elements 160, 176, 192, 208, 224
        block[48] = TQ1_PACKED; // qh run:  elements 240, 244, 248, 252
        let mut output = [0.0f32; 256];
        dequant_tq1_0(&block, &mut output);

        for (level, &trit) in TQ1_TRITS.iter().enumerate() {
            let i0 = level * 32;
            assert_eq!(output[i0], 2.0 * trit, "qs0 level {level} (element {i0})");
            let i1 = 160 + level * 16;
            assert_eq!(output[i1], 2.0 * trit, "qs1 level {level} (element {i1})");
        }
        // The qh run carries only four levels, not five.
        for (level, &trit) in TQ1_TRITS.iter().take(4).enumerate() {
            let ih = 240 + level * 4;
            assert_eq!(output[ih], 2.0 * trit, "qh level {level} (element {ih})");
        }
    }
}
