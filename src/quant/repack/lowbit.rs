//! Repacking between PTQ1_0 and PQ2_0.
//!
//! PTQ1_0 trit `t` and PQ2_0 code `c` both mean the value `(code - 1) * d`
//! at the same 128-element block size, so a repack copies each code and the
//! f16 `d` bytes. No value is rounded. PQ2_0 code 3 means `2 * d`, which
//! ternary cannot hold, so that direction refuses any block holding it.

use crate::error::{Error, Result};
use crate::quant::QuantFormat;
use crate::quant::cpu::kernels::lowbit_codec::{
    LOWBIT_BLOCK_SIZE, PQ2_0_BLOCK_BYTES, PQ2_0_D_OFFSET, PTQ1_0_BLOCK_BYTES, PTQ1_0_D_OFFSET,
    decode_pq2_0_codes, decode_ptq1_0_codes, encode_pq2_0_block, encode_ptq1_0_block,
};

/// Checks `src` holds exactly the blocks of `n_elems` elements in `format`.
///
/// Returns the block count.
fn block_count(format: QuantFormat, src: &[u8], n_elems: usize) -> Result<usize> {
    let expected = format.storage_bytes(n_elems)?;
    if src.len() != expected {
        return Err(Error::QuantError {
            reason: format!(
                "{}: {} bytes for {} elements, expected {}",
                format.name(),
                src.len(),
                n_elems,
                expected
            ),
        });
    }
    Ok(n_elems / LOWBIT_BLOCK_SIZE)
}

/// Repacks PTQ1_0 blocks into PQ2_0 blocks.
///
/// # Errors
/// `n_elems` is not a whole number of blocks, or `src` is not the byte
/// count PTQ1_0 stores for it.
pub fn repack_ptq1_0_to_pq2_0(src: &[u8], n_elems: usize) -> Result<Vec<u8>> {
    let blocks = block_count(QuantFormat::PTQ1_0, src, n_elems)?;
    let mut out = Vec::with_capacity(blocks * PQ2_0_BLOCK_BYTES);
    for block in src.as_chunks::<PTQ1_0_BLOCK_BYTES>().0 {
        let codes = decode_ptq1_0_codes(block);
        let d = [block[PTQ1_0_D_OFFSET], block[PTQ1_0_D_OFFSET + 1]];
        out.extend_from_slice(&encode_pq2_0_block(&codes, d));
    }
    Ok(out)
}

/// Repacks PQ2_0 blocks into PTQ1_0 blocks.
///
/// # Errors
/// `n_elems` is not a whole number of blocks, `src` is not the byte count
/// PQ2_0 stores for it, or a block holds code 3. The error names the block
/// index within `src`.
pub fn repack_pq2_0_to_ptq1_0(src: &[u8], n_elems: usize) -> Result<Vec<u8>> {
    let blocks = block_count(QuantFormat::PQ2_0, src, n_elems)?;
    let mut out = Vec::with_capacity(blocks * PTQ1_0_BLOCK_BYTES);
    for (index, block) in src.as_chunks::<PQ2_0_BLOCK_BYTES>().0.iter().enumerate() {
        let codes = decode_pq2_0_codes(block);
        let d = [block[PQ2_0_D_OFFSET], block[PQ2_0_D_OFFSET + 1]];
        let packed = encode_ptq1_0_block(&codes, d).ok_or_else(|| Error::QuantError {
            reason: format!(
                "PQ2_0 block {index} holds code 3 (value 2*d), which ternary PTQ1_0 cannot \
                 represent"
            ),
        })?;
        out.extend_from_slice(&packed);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant_lowbit::{dequant_pq2_0, dequant_ptq1_0};
    use half::f16;

    const BLOCKS: usize = 16;
    const N: usize = BLOCKS * LOWBIT_BLOCK_SIZE;

    /// PTQ1_0 bytes for `BLOCKS` blocks of pseudo-random trits, each block
    /// under its own scale.
    fn random_ptq1_0(seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 24
        };
        let mut bytes = Vec::with_capacity(BLOCKS * PTQ1_0_BLOCK_BYTES);
        for b in 0..BLOCKS {
            let codes: [u8; LOWBIT_BLOCK_SIZE] = std::array::from_fn(|_| (next() % 3) as u8);
            let d = f16::from_f32(0.5 + b as f32 * 0.125).to_le_bytes();
            let block = encode_ptq1_0_block(&codes, d).expect("ternary codes pack");
            bytes.extend_from_slice(&block);
        }
        bytes
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn ptq1_0_to_pq2_0_and_back_is_byte_identical() {
        for seed in [1, 2, 3] {
            let ptq = random_ptq1_0(seed);
            let pq = repack_ptq1_0_to_pq2_0(&ptq, N).expect("repacks");
            assert_eq!(pq.len(), BLOCKS * PQ2_0_BLOCK_BYTES);
            let back = repack_pq2_0_to_ptq1_0(&pq, N).expect("repacks back");
            assert_eq!(back, ptq, "seed {seed}");
        }
    }

    /// The two kernels decode the source and the repacked blocks to the same
    /// f32 bit patterns.
    #[test]
    fn dequant_agrees_bit_for_bit() {
        let ptq = random_ptq1_0(9);
        let pq = repack_ptq1_0_to_pq2_0(&ptq, N).expect("repacks");

        let mut from_ptq = vec![0.0f32; N];
        dequant_ptq1_0(&ptq, &mut from_ptq);
        let mut from_pq = vec![0.0f32; N];
        dequant_pq2_0(&pq, &mut from_pq);

        assert_eq!(bits(&from_ptq), bits(&from_pq));
        assert!(from_ptq.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn code_3_is_refused_by_block_index() {
        let ptq = random_ptq1_0(4);
        let mut pq = repack_ptq1_0_to_pq2_0(&ptq, N).expect("repacks");
        // Element 0 of block 5 becomes code 3.
        pq[5 * PQ2_0_BLOCK_BYTES + 2] |= 0x03;
        let err = repack_pq2_0_to_ptq1_0(&pq, N)
            .expect_err("code 3 has no trit")
            .to_string();
        assert!(err.contains("block 5"), "{err}");
        assert!(err.contains("code 3"), "{err}");
    }

    #[test]
    fn a_wrong_byte_count_is_refused() {
        let ptq = random_ptq1_0(4);
        assert!(repack_ptq1_0_to_pq2_0(&ptq[..ptq.len() - 1], N).is_err());
        assert!(repack_ptq1_0_to_pq2_0(&ptq, N + 1).is_err());
        assert!(repack_pq2_0_to_ptq1_0(&ptq, N).is_err());
    }
}
