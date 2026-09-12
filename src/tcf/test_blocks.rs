//! Test fixtures for block-encoded tensors: a deterministic Q8_0 stream and
//! a decoder that reads it the way `ggml-common.h` lays it out (`f16 d`,
//! then 32 `i8` codes; value `d * q`). Enough to exercise the container's
//! digests and proof plumbing; the real decoders live with the kernels.

use crate::tcf::binary16::{bits_to_f32, f32_to_bits};
use crate::tcf::encoding::BlockEncoding;
use crate::tcf::error::TcfError;
use crate::tcf::proof::{BlockDecoder, proof_indices};

/// Bytes of one Q8_0 block.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// A Q8_0 stream for a `[rows, cols]` tensor: block `b` (row-major) has
/// scale `0.5 + (b + seed) % 7` and codes `-16 + (b % 5) + i`.
pub fn q8_0_stream(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0, "a Q8_0 row is whole blocks");
    let blocks = rows * cols / 32;
    let mut bytes = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
    for b in 0..blocks {
        let d = f32_to_bits(0.5 + ((b + seed) % 7) as f32);
        bytes.extend_from_slice(&d.to_le_bytes());
        for i in 0..32i32 {
            let q = -16 + (b % 5) as i32 + i;
            bytes.push(q as i8 as u8);
        }
    }
    bytes
}

/// The value at row-major `index` of a Q8_0 stream over `cols`-wide rows.
pub fn q8_0_value(payload: &[u8], cols: usize, index: usize) -> f32 {
    let block = index / 32;
    let base = block * Q8_0_BLOCK_BYTES;
    let d = bits_to_f32(u16::from_le_bytes([payload[base], payload[base + 1]]));
    let q = payload[base + 2 + index % 32] as i8;
    let _ = cols;
    d * f32::from(q)
}

/// The proof values a producer of `payload` would write, as f32.
pub fn q8_0_proof(payload: &[u8], dims: &[u64], tensor_id: u32) -> Vec<f32> {
    let cols = dims[dims.len() - 1] as usize;
    proof_indices(dims, dims.len() as u32, tensor_id)
        .expect("proof indices")
        .iter()
        .map(|&i| q8_0_value(payload, cols, i as usize))
        .collect()
}

/// A reader-side decoder. `bias` is added to every value, so a second
/// decoder can disagree with the producer's.
pub struct Q8_0Decoder {
    pub bias: f32,
}

impl BlockDecoder for Q8_0Decoder {
    fn values_at(
        &self,
        encoding: BlockEncoding,
        payload: &[u8],
        dims: &[u64],
        _rank: u32,
        _tensor_id: u32,
        indices: &[u64],
    ) -> Result<Vec<f32>, TcfError> {
        assert_eq!(encoding, BlockEncoding::Q8_0);
        let cols = dims[dims.len() - 1] as usize;
        Ok(indices
            .iter()
            .map(|&i| q8_0_value(payload, cols, i as usize) + self.bias)
            .collect())
    }
}
