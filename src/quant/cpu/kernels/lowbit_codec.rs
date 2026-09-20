//! Block code codecs for the lowbit formats PQ2_0 and PTQ1_0.
//!
//! `dequant_lowbit` turns a block into f32. This file turns a block into
//! its integer codes and back, so the two formats can be repacked without a
//! float round trip. The layout constants live here once; the dequant
//! kernels read them from here.
//!
//! ```text
//!   PQ2_0  (34B): d[0..2],  qs[2..34]              2-bit code per element
//!   PTQ1_0 (28B): qs[0..24], qh[24..26], d[26..28]  5 or 4 trits per byte
//! ```
//!
//! A code is the stored integer: PQ2_0 code `c` in 0..4 and PTQ1_0 trit
//! `t` in 0..3 both mean the value `(code - 1) * d`.

use super::dequant_lowbit::{code2, ptq1_0_slot};
use super::dequant_tq::base3_trit;

/// Elements per block, shared by PQ2_0 and PTQ1_0.
pub const LOWBIT_BLOCK_SIZE: usize = 128;

/// PQ2_0 block: `d:f16 + qs[32]`.
pub const PQ2_0_BLOCK_BYTES: usize = 34;
/// Byte offset of the f16 scale in a PQ2_0 block.
pub const PQ2_0_D_OFFSET: usize = 0;
/// Byte offset of the 2-bit codes in a PQ2_0 block.
pub const PQ2_0_QS_OFFSET: usize = 2;

/// PTQ1_0 block: `qs[24] + qh[2] + d:f16`.
pub const PTQ1_0_BLOCK_BYTES: usize = 28;
/// Byte offset of the f16 scale in a PTQ1_0 block.
pub const PTQ1_0_D_OFFSET: usize = 26;
/// Packed trit bytes in a PTQ1_0 block: `qs[24] + qh[2]`.
const PTQ1_0_PACKED_BYTES: usize = 26;

/// Packs five trit codes (0..3 each, the first most significant) as
/// llama.cpp's `quantize_row_tq1_0_ref` does: base 3, then a ceiling scale
/// by 256/243 so `base3_trit` recovers each digit with a wrapping multiply.
pub fn base3_pack5(codes: [u8; 5]) -> u8 {
    let q = codes.iter().fold(0u16, |q, &c| q * 3 + u16::from(c));
    (q * 256).div_ceil(243) as u8
}

/// The 128 codes of one PQ2_0 block, each in 0..4.
pub fn decode_pq2_0_codes(block: &[u8]) -> [u8; LOWBIT_BLOCK_SIZE] {
    std::array::from_fn(|elem| code2(&block[PQ2_0_QS_OFFSET..], elem))
}

/// The 128 trit codes of one PTQ1_0 block, each in 0..3.
pub fn decode_ptq1_0_codes(block: &[u8]) -> [u8; LOWBIT_BLOCK_SIZE] {
    std::array::from_fn(|elem| {
        let (byte, level) = ptq1_0_slot(elem);
        (base3_trit(block[byte], level) + 1) as u8
    })
}

/// One PQ2_0 block from 128 codes (0..4) and the f16 scale bytes.
pub fn encode_pq2_0_block(codes: &[u8; LOWBIT_BLOCK_SIZE], d: [u8; 2]) -> [u8; PQ2_0_BLOCK_BYTES] {
    let mut block = [0u8; PQ2_0_BLOCK_BYTES];
    block[PQ2_0_D_OFFSET..PQ2_0_D_OFFSET + 2].copy_from_slice(&d);
    for (elem, &code) in codes.iter().enumerate() {
        block[PQ2_0_QS_OFFSET + elem / 4] |= (code & 0x03) << ((elem % 4) * 2);
    }
    block
}

/// One PTQ1_0 block from 128 trit codes and the f16 scale bytes.
///
/// `None` when any code is above 2: ternary holds -1, 0 and +1 only.
pub fn encode_ptq1_0_block(
    codes: &[u8; LOWBIT_BLOCK_SIZE],
    d: [u8; 2],
) -> Option<[u8; PTQ1_0_BLOCK_BYTES]> {
    if codes.iter().any(|&code| code > 2) {
        return None;
    }
    // The fifth digit of a `qh` byte stays 0: those bytes hold four trits.
    let mut digits = [[0u8; 5]; PTQ1_0_PACKED_BYTES];
    for (elem, &code) in codes.iter().enumerate() {
        let (byte, level) = ptq1_0_slot(elem);
        digits[byte][level] = code;
    }
    let mut block = [0u8; PTQ1_0_BLOCK_BYTES];
    for (byte, packed) in digits.iter().enumerate() {
        block[byte] = base3_pack5(*packed);
    }
    block[PTQ1_0_D_OFFSET..PTQ1_0_D_OFFSET + 2].copy_from_slice(&d);
    Some(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic codes in 0..`modulus` for one block.
    fn codes(seed: u32, modulus: u8) -> [u8; LOWBIT_BLOCK_SIZE] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 24) as u8) % modulus
        })
    }

    /// Every one of the 243 five-trit bytes decodes digit by digit.
    #[test]
    fn base3_pack5_round_trips_every_byte() {
        for q in 0u16..243 {
            let digits: [u8; 5] = std::array::from_fn(|i| ((q / 3u16.pow(4 - i as u32)) % 3) as u8);
            let byte = base3_pack5(digits);
            for (level, &digit) in digits.iter().enumerate() {
                assert_eq!(
                    base3_trit(byte, level) + 1,
                    i32::from(digit),
                    "q {q} level {level}"
                );
            }
        }
    }

    #[test]
    fn pq2_0_codes_round_trip() {
        let codes = codes(7, 4);
        let block = encode_pq2_0_block(&codes, [0x34, 0x12]);
        assert_eq!(&block[..2], &[0x34, 0x12]);
        assert_eq!(decode_pq2_0_codes(&block), codes);
    }

    #[test]
    fn ptq1_0_codes_round_trip() {
        let codes = codes(11, 3);
        let block = encode_ptq1_0_block(&codes, [0x34, 0x12]).expect("ternary codes pack");
        assert_eq!(&block[26..], &[0x34, 0x12]);
        assert_eq!(decode_ptq1_0_codes(&block), codes);
    }

    /// Encoding is canonical: decode then encode gives the same bytes.
    #[test]
    fn ptq1_0_encode_is_the_inverse_of_decode() {
        let codes = codes(3, 3);
        let block = encode_ptq1_0_block(&codes, [1, 2]).expect("ternary codes pack");
        let again =
            encode_ptq1_0_block(&decode_ptq1_0_codes(&block), [1, 2]).expect("decoded codes pack");
        assert_eq!(block, again);
    }

    #[test]
    fn ptq1_0_refuses_code_3() {
        let mut codes = codes(5, 3);
        codes[127] = 3;
        assert!(encode_ptq1_0_block(&codes, [0, 0]).is_none());
    }
}
