//! Proof vectors: 64 spot-check values a reader can screen a quantized
//! tensor against before paying to hash its payload. See `FORMAT.md`.
//!
//! A proof vector is a lazy screen, never the authority: the digests decide.
//! For a block-encoded tensor the values come from the reader's own block
//! decoder, so a file whose producer decoded the blocks differently from
//! this runtime fails here rather than at first use.

use crate::tcf::binary16::f32_to_bits;
use crate::tcf::consts::PROOF_COUNT;
use crate::tcf::encoding::block::BlockEncoding;
use crate::tcf::error::TcfError;

/// `proof_format` of a raw tensor: no proof vector. Section 15.3.
///
/// This is the one enumerated field where `0` is a valid stored value. A
/// reader MUST accept it on a raw tensor and reject it on a quantized one.
pub const PROOF_FORMAT_NONE: u32 = 0;

/// `proof_format` of a quantized tensor: one LE binary16 expected
/// dequantized value per proof index. Section 15.3.
pub const PROOF_FORMAT_F16: u32 = 1;

/// Bytes a quantized tensor's proof vector occupies: `PROOF_COUNT` binary16
/// values. Section 15.3.
pub const PROOF_BYTES: usize = 128;

/// Proof indices spanning `dims`, per Section 15.3.
///
/// ```text
/// proof entries 0 through 15 inclusive    index = the entry number
/// proof entries 16 through 63 inclusive   r = entry number - 15, r runs 1 through 48
///                                         index = 15 + floor(r * (N - 16) / 48)
/// ```
///
/// `N = product(dims)`. The first 16 indices cover the tensor's opening
/// elements, where a packing error shows up first; the remaining 48 spread
/// evenly across the rest, and the last one is always `N - 1`, so the final
/// element is always probed.
///
/// Returns exactly [`PROOF_COUNT`] indices, non-decreasing, every one below
/// `N`.
///
/// `tensor_id` names the tensor in any error this returns; it takes no part
/// in the formula.
///
/// # Errors
/// - [`TcfError::InvalidRank`] if `rank` is outside `1..=8`, or
///   `dims.len() != rank`.
/// - [`TcfError::TileArithmeticOverflow`] if `product(dims)` overflows `u64`.
/// - [`TcfError::InvalidQuantShape`] if `N < 64`: the formula is defined for
///   `N >= 64` only.
pub fn proof_indices(dims: &[u64], rank: u32, tensor_id: u32) -> Result<Vec<u64>, TcfError> {
    let n = element_count(dims, rank)?;
    let count = u64::from(PROOF_COUNT);
    if n < count {
        return Err(TcfError::InvalidQuantShape { tensor_id });
    }
    // The head is the first 16 element indices verbatim; the tail spreads
    // the remaining 48 over `16..N`.
    let head: u64 = 16;
    let tail = count
        .checked_sub(head)
        .ok_or(TcfError::InvalidQuantShape { tensor_id })?;
    let span = u128::from(
        n.checked_sub(head)
            .ok_or(TcfError::InvalidQuantShape { tensor_id })?,
    );

    let mut indices = Vec::with_capacity(PROOF_COUNT as usize);
    for j in 0..head {
        indices.push(j);
    }
    for r in 1..=tail {
        // Widened to u128 so `r * (N - 16)` cannot wrap for any u64 `N`.
        let scaled = u128::from(r)
            .checked_mul(span)
            .ok_or(TcfError::TileArithmeticOverflow)?
            .checked_div(u128::from(tail))
            .ok_or(TcfError::TileArithmeticOverflow)?;
        let index = u64::try_from(scaled)
            .map_err(|_| TcfError::TileArithmeticOverflow)?
            .checked_add(head.saturating_sub(1))
            .ok_or(TcfError::TileArithmeticOverflow)?;
        indices.push(index);
    }
    Ok(indices)
}

/// A decoder for GGML block payloads, supplied by whoever owns the block
/// kernels. TCF carries block streams without decoding them, so the proof
/// values of a block-encoded tensor come from — and are checked against —
/// the caller's decoder.
pub trait BlockDecoder {
    /// The dequantized value at each row-major `index` of the tensor whose
    /// block stream is `payload`, in f32. `tensor_id` labels any error.
    ///
    /// # Errors
    /// Whatever the decoder raises for a payload it cannot read; the caller
    /// reports it against the tensor.
    fn values_at(
        &self,
        encoding: BlockEncoding,
        payload: &[u8],
        dims: &[u64],
        rank: u32,
        tensor_id: u32,
        indices: &[u64],
    ) -> Result<Vec<f32>, TcfError>;
}

/// Proof values for a block-encoded tensor, as binary16 bits: the decoder's
/// value at each proof index, rounded to binary16.
///
/// # Errors
/// Every error [`proof_indices`] returns; [`TcfError::InvalidQuantShape`] if
/// the decoder returns the wrong count; the decoder's own errors.
pub fn block_proof_values(
    decoder: &dyn BlockDecoder,
    encoding: BlockEncoding,
    payload: &[u8],
    dims: &[u64],
    rank: u32,
    tensor_id: u32,
) -> Result<Vec<u16>, TcfError> {
    let indices = proof_indices(dims, rank, tensor_id)?;
    let values = decoder.values_at(encoding, payload, dims, rank, tensor_id, &indices)?;
    if values.len() != indices.len() {
        return Err(TcfError::InvalidQuantShape { tensor_id });
    }
    Ok(values.into_iter().map(f32_to_bits).collect())
}

/// `N = product(dims)`. Section 15.3.
fn element_count(dims: &[u64], rank: u32) -> Result<u64, TcfError> {
    let rank_usize = usize::try_from(rank).map_err(|_| TcfError::InvalidRank { rank })?;
    if !(1..=crate::tcf::consts::MAX_RANK).contains(&rank) || dims.len() != rank_usize {
        return Err(TcfError::InvalidRank { rank });
    }
    let mut n: u64 = 1;
    for &dim in dims {
        n = n.checked_mul(dim).ok_or(TcfError::TileArithmeticOverflow)?;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Section 15.3, spelled out independently of the implementation.
    fn reference_indices(n: u64) -> Vec<u64> {
        let mut out: Vec<u64> = (0..16).collect();
        for j in 16..64u64 {
            let r = j - 15;
            out.push(15 + (u128::from(r) * u128::from(n - 16) / 48) as u64);
        }
        out
    }

    #[test]
    fn indices_match_the_section_15_3_formula() {
        for n in [64u64, 4096, 1_000_000, 70_000_000_000] {
            let indices = proof_indices(&[n / 64, 64], 2, 0).unwrap();
            assert_eq!(indices, reference_indices(n), "N = {n}");
        }
    }

    #[test]
    fn indices_are_sixty_four_in_range_non_decreasing_and_end_at_n_minus_one() {
        for n in [64u64, 4096, 262_144, 70_000_000_000] {
            let indices = proof_indices(&[n / 64, 64], 2, 0).unwrap();
            assert_eq!(indices.len(), PROOF_COUNT as usize, "N = {n}");
            assert!(indices.windows(2).all(|w| w[0] <= w[1]), "N = {n}");
            assert!(indices.iter().all(|&i| i < n), "N = {n}");
            assert_eq!(*indices.last().unwrap(), n - 1, "N = {n}");
            assert_eq!(&indices[..16], &(0..16).collect::<Vec<u64>>()[..]);
        }
    }

    /// At exactly one tile the formula degenerates to the identity: every
    /// element is probed.
    #[test]
    fn a_single_tile_probes_every_element() {
        let indices = proof_indices(&[1, 64], 2, 0).unwrap();
        assert_eq!(indices, (0..64).collect::<Vec<u64>>());
    }

    #[test]
    fn a_tensor_smaller_than_one_tile_is_rejected() {
        assert_eq!(
            proof_indices(&[8, 4], 2, 42),
            Err(TcfError::InvalidQuantShape { tensor_id: 42 })
        );
    }

    #[test]
    fn bad_rank_is_rejected() {
        assert_eq!(
            proof_indices(&[1, 64], 3, 0),
            Err(TcfError::InvalidRank { rank: 3 })
        );
        assert_eq!(
            proof_indices(&[64], 0, 0),
            Err(TcfError::InvalidRank { rank: 0 })
        );
    }

    #[test]
    fn proof_format_constants_match_section_15_3() {
        assert_eq!(PROOF_FORMAT_NONE, 0);
        assert_eq!(PROOF_FORMAT_F16, 1);
        assert_eq!(PROOF_BYTES, PROOF_COUNT as usize * 2);
    }
}
