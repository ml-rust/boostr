//! CPU dequantization kernels for the lowbit formats
//!
//! Q1_0, Q2_0, PQ2_0, PTQ1_0
//!
//! Ported from llama.cpp's `ggml-quants.c` (`dequantize_row_q1_0`,
//! `dequantize_row_q2_0`, `dequantize_row_pq2_0`, `dequantize_row_ptq1_0`).
//! Q1_0, Q2_0 and PQ2_0 store the f16 scale `d` at the START of the block
//! and order elements byte-major, low bits first. PTQ1_0 is TQ1_0's base-3
//! trit packing at group 128: `d` at the END, elements level-major over
//! three runs.
//!
//! ```text
//!   Q1_0   (18B): d[0..2],  qs[2..18]              1 bit  each, 128 elements
//!   Q2_0   (18B): d[0..2],  qs[2..18]              2 bits each,  64 elements
//!   PQ2_0  (34B): d[0..2],  qs[2..34]              2 bits each, 128 elements
//!   PTQ1_0 (28B): qs[0..24], qh[24..26], d[26..28]  5 or 4 trits per byte
//! ```
//!
//! The CUDA side keeps the same layouts in `cuda/kernels/decode.cuh`; the
//! two must stay in agreement.
use super::dequant_tq::base3_trit;
use half::f16;

/// Sign value {-1, 1} of element `elem` of a Q1_0 `qs` run.
///
/// Bit set means `+d`, bit clear means `-d`. Eight elements per byte, low
/// bit first.
#[inline]
fn sign_bit(qs: &[u8], elem: usize) -> i32 {
    if (qs[elem / 8] >> (elem % 8)) & 1 == 1 {
        1
    } else {
        -1
    }
}

/// Value {-1, 0, 1, 2} of element `elem` of a Q2_0 or PQ2_0 `qs` run.
///
/// Four 2-bit codes per byte, low bits first. llama.cpp maps code `q` to
/// `q - 1`: `00=-1, 01=0, 10=+1, 11=+2`.
#[inline]
fn code2_minus_1(qs: &[u8], elem: usize) -> i32 {
    i32::from((qs[elem / 4] >> ((elem % 4) * 2)) & 0x03) - 1
}

/// Ternary value {-1, 0, 1} of element `elem` of a PTQ1_0 block.
///
/// The 128 elements come from three differently shaped runs, in this order:
/// `[0, 80)` is `qs[0..16]` over 5 levels, `[80, 120)` is `qs[16..24]` over
/// 5 levels, and `[120, 128)` is `qh[0..2]` over 4 levels. These are the
/// 16-byte and 8-byte stages of llama.cpp's `ptq1_0_stages = {32, 16, 8}`
/// applied to a 24-byte `qs`.
#[inline]
fn ptq1_0_trit(block: &[u8], elem: usize) -> i32 {
    let (byte, level) = if elem < 80 {
        (block[elem % 16], elem / 16)
    } else if elem < 120 {
        let r = elem - 80;
        (block[16 + r % 8], r / 8)
    } else {
        let r = elem - 120;
        (block[24 + r % 2], r / 2)
    };
    base3_trit(byte, level)
}

/// Scales each block's integer codes by its f16 `d`.
///
/// `code(block, elem)` reads the code of one element from the whole block.
#[inline]
fn dequant_blocks(
    blocks: &[u8],
    output: &mut [f32],
    block_size: usize,
    block_bytes: usize,
    d_offset: usize,
    code: impl Fn(&[u8], usize) -> i32,
) {
    let num_blocks = blocks.len() / block_bytes;
    debug_assert_eq!(output.len(), num_blocks * block_size);

    for b in 0..num_blocks {
        let block = &blocks[b * block_bytes..][..block_bytes];
        let d = f16::from_le_bytes([block[d_offset], block[d_offset + 1]]).to_f32();
        let out = &mut output[b * block_size..][..block_size];
        for (elem, slot) in out.iter_mut().enumerate() {
            *slot = d * code(block, elem) as f32;
        }
    }
}

/// Dequantizes Q1_0 blocks to f32
///
/// Q1_0: 128 elements, 18 bytes/block. Layout `d:f16 + qs[16]`.
pub fn dequant_q1_0(blocks: &[u8], output: &mut [f32]) {
    dequant_blocks(blocks, output, 128, 18, 0, |block, elem| {
        sign_bit(&block[2..], elem)
    });
}

/// Dequantizes Q2_0 blocks to f32
///
/// Q2_0: 64 elements, 18 bytes/block. Layout `d:f16 + qs[16]`.
pub fn dequant_q2_0(blocks: &[u8], output: &mut [f32]) {
    dequant_blocks(blocks, output, 64, 18, 0, |block, elem| {
        code2_minus_1(&block[2..], elem)
    });
}

/// Dequantizes PQ2_0 blocks to f32
///
/// PQ2_0: 128 elements, 34 bytes/block. Layout `d:f16 + qs[32]`.
pub fn dequant_pq2_0(blocks: &[u8], output: &mut [f32]) {
    dequant_blocks(blocks, output, 128, 34, 0, |block, elem| {
        code2_minus_1(&block[2..], elem)
    });
}

/// Dequantizes PTQ1_0 blocks to f32
///
/// PTQ1_0: 128 elements, 28 bytes/block. Layout `qs[24] + qh[2] + d:f16`.
pub fn dequant_ptq1_0(blocks: &[u8], output: &mut [f32]) {
    dequant_blocks(blocks, output, 128, 28, 26, ptq1_0_trit);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0b00_01_10_11` — 2-bit fields 0..4 are 3, 2, 1, 0, i.e. values
    /// 2, 1, 0, -1 at four adjacent elements.
    const CODE2_PACKED: u8 = 0x1B;
    const CODE2_VALUES: [f32; 4] = [2.0, 1.0, 0.0, -1.0];

    /// `0b1011_0001` — bits 0..8 are 1, 0, 0, 0, 1, 1, 0, 1.
    const SIGN_PACKED: u8 = 0xB1;
    const SIGN_VALUES: [f32; 8] = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0];

    fn d_bytes(d: f32) -> [u8; 2] {
        f16::from_f32(d).to_le_bytes()
    }

    /// Packs five trits the way `quantize_row_ptq1_0_ref` does: base 3,
    /// first trit most significant, then a ceiling scale by 256/243.
    fn pack5(trits: [i32; 5]) -> u8 {
        let q = trits.iter().fold(0u16, |q, &t| q * 3 + (t + 1) as u16);
        (q * 256).div_ceil(243) as u8
    }

    /// Packs four trits for `qh`: the first trit lands in the most
    /// significant position of a five-trit byte, the fifth slot is zero.
    fn pack4(trits: [i32; 4]) -> u8 {
        pack5([trits[0], trits[1], trits[2], trits[3], -1])
    }

    /// Builds one PTQ1_0 block from 128 trits by llama.cpp's packing rule.
    fn pack_ptq1_0(trits: &[i32; 128], d: f32) -> [u8; 28] {
        let mut block = [0u8; 28];
        for (m, byte) in block[0..16].iter_mut().enumerate() {
            *byte = pack5(std::array::from_fn(|n| trits[m + n * 16]));
        }
        for (m, byte) in block[16..24].iter_mut().enumerate() {
            *byte = pack5(std::array::from_fn(|n| trits[80 + m + n * 8]));
        }
        for (h, byte) in block[24..26].iter_mut().enumerate() {
            *byte = pack4(std::array::from_fn(|n| trits[120 + h + n * 2]));
        }
        block[26..28].copy_from_slice(&d_bytes(d));
        block
    }

    #[test]
    fn test_dequant_q1_0_layout() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&d_bytes(2.0));
        block[2] = SIGN_PACKED; // elements 0..8
        block[17] = SIGN_PACKED; // elements 120..128
        let mut output = [0.0f32; 128];
        dequant_q1_0(&block, &mut output);

        for (i, &v) in SIGN_VALUES.iter().enumerate() {
            assert_eq!(output[i], 2.0 * v, "element {i}");
            assert_eq!(output[120 + i], 2.0 * v, "element {}", 120 + i);
        }
        // A clear bit is -d, never 0.
        assert_eq!(output[8], -2.0);
    }

    #[test]
    fn test_dequant_q1_0_all_set() {
        let mut block = [0xFFu8; 18];
        block[0..2].copy_from_slice(&d_bytes(1.0));
        let mut output = [0.0f32; 128];
        dequant_q1_0(&block, &mut output);
        assert!(output.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_dequant_q2_0_layout() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&d_bytes(2.0));
        block[2] = CODE2_PACKED; // elements 0..4
        block[17] = CODE2_PACKED; // elements 60..64
        let mut output = [0.0f32; 64];
        dequant_q2_0(&block, &mut output);

        for (i, &v) in CODE2_VALUES.iter().enumerate() {
            assert_eq!(output[i], 2.0 * v, "element {i}");
            assert_eq!(output[60 + i], 2.0 * v, "element {}", 60 + i);
        }
        // Code 0 is -d, never 0.
        assert_eq!(output[4], -2.0);
    }

    #[test]
    fn test_dequant_pq2_0_layout() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&d_bytes(2.0));
        block[2] = CODE2_PACKED; // elements 0..4
        block[33] = CODE2_PACKED; // elements 124..128
        let mut output = [0.0f32; 128];
        dequant_pq2_0(&block, &mut output);

        for (i, &v) in CODE2_VALUES.iter().enumerate() {
            assert_eq!(output[i], 2.0 * v, "element {i}");
            assert_eq!(output[124 + i], 2.0 * v, "element {}", 124 + i);
        }
        assert_eq!(output[4], -2.0);
    }

    /// Code `01` in every field is 0; code `10` in every field is +1.
    #[test]
    fn test_dequant_pq2_0_uniform_codes() {
        for (fill, expect) in [(0x55u8, 0.0f32), (0xAA, 1.0), (0x00, -1.0), (0xFF, 2.0)] {
            let mut block = [fill; 34];
            block[0..2].copy_from_slice(&d_bytes(1.0));
            let mut output = [7.0f32; 128];
            dequant_pq2_0(&block, &mut output);
            assert!(
                output.iter().all(|&v| v == expect),
                "fill {fill:#04x} expected {expect}"
            );
        }
    }

    /// Pins all three runs, the `d` offset, and the last trit of a `qs`
    /// byte by round-tripping a pattern that hits every trit value.
    #[test]
    fn test_dequant_ptq1_0_layout() {
        let trits: [i32; 128] = std::array::from_fn(|i| ((i * 7 + 3) % 3) as i32 - 1);
        let block = pack_ptq1_0(&trits, 2.0);
        let mut output = [0.0f32; 128];
        dequant_ptq1_0(&block, &mut output);

        for (i, (&got, &t)) in output.iter().zip(&trits).enumerate() {
            assert_eq!(got, 2.0 * t as f32, "element {i}");
        }
        // Level 4 of qs[0] is element 64; level 4 of qs[16] is element 112.
        assert_eq!(output[64], 2.0 * trits[64] as f32);
        assert_eq!(output[112], 2.0 * trits[112] as f32);
        // The qh tail: qh[1] level 3 is the last element.
        assert_eq!(output[127], 2.0 * trits[127] as f32);
    }

    #[test]
    fn test_dequant_ptq1_0_all_zero_trits() {
        let block = pack_ptq1_0(&[0; 128], 1.0);
        assert_eq!(block[0], 128); // packed [0,0,0,0,0]
        let mut output = [7.0f32; 128];
        dequant_ptq1_0(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_ptq1_0_all_plus_one() {
        let block = pack_ptq1_0(&[1; 128], 1.0);
        assert_eq!(block[0], 255); // packed [2,2,2,2,2]
        let mut output = [0.0f32; 128];
        dequant_ptq1_0(&block, &mut output);
        assert!(output.iter().all(|&v| v == 1.0));
    }

    /// The `qh` tail decodes only four levels: byte `qh[h]` feeds elements
    /// `120 + h + 2 * level`.
    #[test]
    fn test_dequant_ptq1_0_qh_tail() {
        let mut block = [0u8; 28];
        block[26..28].copy_from_slice(&d_bytes(2.0));
        block[24] = pack4([-1, 0, 1, -1]);
        block[25] = pack4([1, 1, 0, -1]);
        let mut output = [0.0f32; 128];
        dequant_ptq1_0(&block, &mut output);
        let expect = [-1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0];
        for (i, &v) in expect.iter().enumerate() {
            assert_eq!(output[120 + i], 2.0 * v, "element {}", 120 + i);
        }
    }
}
