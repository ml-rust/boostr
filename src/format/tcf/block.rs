//! GGML block encodings inside a TCF file, read with boostr's own kernels.
//!
//! A block-encoded TCF tensor stores the same byte stream a GGUF file stores
//! for the same `ggml_type` ([`BlockEncoding`]). TCF never restates
//! the block layout, so the layout knowledge stays where it already lives:
//! [`QuantFormat`] names it and the CPU dequant kernels decode it. This
//! module is the bridge — encoding to format, stream to values — and holds no
//! bit position of its own.
//!
//! The proof vector of a block tensor is checked with [`BoostrBlockDecoder`],
//! the reader-side decoder [`crate::tcf`] asks for. A file written by a producer
//! whose decoder disagrees with these kernels fails verification here, which
//! is the point: the proof binds the bytes to the values this runtime will
//! compute from them.

use crate::error::{Error, Result};
use crate::quant::QuantFormat;
use crate::quant::cpu::kernels::quant_matmul::dequant_row_f32;
use crate::tcf::{BlockDecoder, BlockEncoding, TcfError, TensorRecord};

/// The runtime format for a block encoding.
///
/// # Errors
/// [`Error::ModelError`] when boostr has no kernel for the `ggml_type`.
pub fn block_format(block: BlockEncoding, name: &str) -> Result<QuantFormat> {
    QuantFormat::from_ggml_type_id(u32::from(block.ggml_type())).map_err(|_| Error::ModelError {
        reason: format!(
            "TCF tensor '{name}': block encoding {} (ggml_type {}) has no kernel in this build",
            block.name(),
            block.ggml_type()
        ),
    })
}

/// Row geometry of a block stream: elements per row and bytes per row.
struct RowGeometry {
    rows: usize,
    row_elems: usize,
    row_bytes: usize,
}

/// Split `dims` into rows of the last axis and size each row's block bytes.
fn row_geometry(format: QuantFormat, dims: &[u64], payload_len: usize) -> Option<RowGeometry> {
    let (last, lead) = dims.split_last()?;
    let row_elems = usize::try_from(*last).ok()?;
    let mut rows: usize = 1;
    for dim in lead {
        rows = rows.checked_mul(usize::try_from(*dim).ok()?)?;
    }
    let row_bytes = format.storage_bytes(row_elems).ok()?;
    if rows.checked_mul(row_bytes)? != payload_len {
        return None;
    }
    Some(RowGeometry {
        rows,
        row_elems,
        row_bytes,
    })
}

/// Decode a whole block stream to row-major f32 values.
///
/// # Errors
/// [`Error::ModelError`] when the encoding has no kernel or the payload
/// length disagrees with the shape.
pub fn decode_block_f32(
    block: BlockEncoding,
    record: &TensorRecord,
    payload: &[u8],
    name: &str,
) -> Result<Vec<f32>> {
    let format = block_format(block, name)?;
    let geometry =
        row_geometry(format, record.shape(), payload.len()).ok_or_else(|| Error::ModelError {
            reason: format!(
                "TCF tensor '{name}': {} bytes of {} blocks do not cover shape {:?}",
                payload.len(),
                format.name(),
                record.shape()
            ),
        })?;
    let mut out = vec![0.0f32; geometry.rows * geometry.row_elems];
    for (row_bytes, row_out) in payload
        .chunks_exact(geometry.row_bytes)
        .zip(out.chunks_exact_mut(geometry.row_elems))
    {
        dequant_row_f32(row_bytes, row_out, format);
    }
    Ok(out)
}

/// The reader-side [`BlockDecoder`], backed by the CPU dequant kernels.
///
/// The proof indices are a fixed handful per tensor, so this decodes only the
/// rows they land in, each row once.
pub struct BoostrBlockDecoder;

impl BlockDecoder for BoostrBlockDecoder {
    fn values_at(
        &self,
        encoding: BlockEncoding,
        payload: &[u8],
        dims: &[u64],
        _rank: u32,
        tensor_id: u32,
        indices: &[u64],
    ) -> std::result::Result<Vec<f32>, TcfError> {
        let format =
            QuantFormat::from_ggml_type_id(u32::from(encoding.ggml_type())).map_err(|_| {
                TcfError::UnsupportedEncoding {
                    raw: encoding.to_u16(),
                }
            })?;
        let geometry = row_geometry(format, dims, payload.len())
            .ok_or(TcfError::InvalidQuantShape { tensor_id })?;
        let mut row_values = vec![0.0f32; geometry.row_elems];
        let mut decoded_row: Option<usize> = None;
        let mut out = Vec::with_capacity(indices.len());
        for &index in indices {
            let index =
                usize::try_from(index).map_err(|_| TcfError::InvalidQuantShape { tensor_id })?;
            let row = index / geometry.row_elems;
            let col = index % geometry.row_elems;
            if row >= geometry.rows {
                return Err(TcfError::InvalidQuantShape { tensor_id });
            }
            if decoded_row != Some(row) {
                let start = row * geometry.row_bytes;
                dequant_row_f32(
                    &payload[start..start + geometry.row_bytes],
                    &mut row_values,
                    format,
                );
                decoded_row = Some(row);
            }
            out.push(row_values[col]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rows of 64 as Q8_0: `f16 d` then 32 `i8` codes per block.
    fn q8_0_stream() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 * 34);
        for block in 0..4u8 {
            let d = half::f16::from_f32(0.5 + f32::from(block)).to_bits();
            bytes.extend_from_slice(&d.to_le_bytes());
            for code in 0..32u8 {
                bytes.push((code as i8 - 16 + block as i8) as u8);
            }
        }
        bytes
    }

    #[test]
    fn every_block_encoding_maps_to_a_format() {
        for block in BlockEncoding::ALL {
            let format = block_format(block, "t").expect("format");
            assert_eq!(format.ggml_type_id(), u32::from(block.ggml_type()));
            assert_eq!(format.block_size() as u64, block.block_elems());
            assert_eq!(format.block_bytes() as u64, block.block_bytes());
        }
    }

    #[test]
    fn sampled_values_match_the_full_decode() {
        let bytes = q8_0_stream();
        let indices = crate::tcf::proof_indices(&[2, 64], 2, 3).expect("indices");
        let sampled = BoostrBlockDecoder
            .values_at(BlockEncoding::Q8_0, &bytes, &[2, 64], 2, 3, &indices)
            .expect("decodes");
        let mut full = vec![0.0f32; 128];
        for (row, out) in bytes
            .as_chunks::<68>()
            .0
            .iter()
            .zip(full.as_chunks_mut::<64>().0.iter_mut())
        {
            dequant_row_f32(row, out, QuantFormat::Q8_0);
        }
        for (i, v) in indices.iter().zip(&sampled) {
            assert_eq!(*v, full[*i as usize]);
        }
        // Block 1 code 0: d = 1.5, q = -15.
        assert_eq!(full[32], -22.5);
    }

    #[test]
    fn a_short_stream_is_rejected() {
        let bytes = q8_0_stream();
        let err = BoostrBlockDecoder
            .values_at(BlockEncoding::Q8_0, &bytes[..100], &[2, 64], 2, 3, &[0])
            .expect_err("short");
        assert_eq!(err, TcfError::InvalidQuantShape { tensor_id: 3 });
    }
}
