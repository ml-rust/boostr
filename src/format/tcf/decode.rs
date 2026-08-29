//! One tensor's payload bytes to dense row-major f32 values.
//!
//! Native quantized encodings decode through `tcf-core`'s reference codec:
//! `unpack` rebuilds the logical tiles, and
//! [`crate::quant::cpu::kernels::tcf::dequantize_tiles_into`] applies
//! Section 13.0 or Section 13.0.1 with a vector element loop that is
//! bit-identical to `tcf_core::dequantize`. Nothing here holds a bit
//! position or a plane order — a second copy of a block layout is exactly
//! what this format exists to prevent (MIGRATION.md Section 4.5.3).
//!
//! Raw encodings convert element by element through numr's dtype
//! conversions. Section 12: a raw encoding stores literal values with no
//! scale of any kind.

use crate::error::{Error, Result};
use numr::dtype::{FP8E4M3, FP8E5M2};
use tcf_core::{Encoding, RawEncoding, TensorRecord, tile_count, unpack};

use crate::quant::cpu::kernels::tcf::dequantize_tiles_into;

use super::error::tcf_tensor_error;
use super::metadata::encoding_name;

/// Element count of a record's logical shape. Section 8.
///
/// # Errors
/// [`Error::ModelError`] if the product overflows `usize`.
pub fn element_count(record: &TensorRecord, name: &str) -> Result<usize> {
    let mut count: usize = 1;
    for dim in record.shape() {
        let dim = usize::try_from(*dim).map_err(|_| Error::ModelError {
            reason: format!("TCF tensor '{name}': dimension {dim} exceeds usize"),
        })?;
        count = count.checked_mul(dim).ok_or_else(|| Error::ModelError {
            reason: format!("TCF tensor '{name}': element count overflows usize"),
        })?;
    }
    Ok(count)
}

/// Decode `payload` into `product(dims)` row-major f32 values.
///
/// `payload` is the tensor's logical bytes, alignment padding excluded — what
/// `TcfFile::payload` returns. The caller verifies digests first: this
/// function decodes, it does not authenticate.
///
/// # Errors
/// - [`Error::ModelError`] naming the encoding and the tensor, when the
///   encoding has no decode path here.
/// - [`Error::ModelError`] carrying the spec's `E_*` code, when the codec
///   rejects the payload.
/// - [`Error::ModelError`] when the decoded length disagrees with the shape.
pub fn decode_tensor_f32(record: &TensorRecord, payload: &[u8], name: &str) -> Result<Vec<f32>> {
    let expected = element_count(record, name)?;
    let values = match record.encoding {
        Encoding::Native(native) => {
            // The layout, never the bare geometry: a `QuantGeometry` converts
            // to a flat scale form, which sizes a two-level payload wrong.
            let layout = native.layout();
            let tiles = tile_count(record.shape(), record.rank, layout.geometry.tile)
                .map_err(|e| tcf_tensor_error(name, "tile count", e))?;
            let logical =
                unpack(payload, tiles, layout).map_err(|e| tcf_tensor_error(name, "unpack", e))?;
            let mut decoded = Vec::new();
            dequantize_tiles_into(&logical, layout, &mut decoded)
                .map_err(|e| tcf_tensor_error(name, "dequantize", e))?;
            decoded
        }
        Encoding::Raw(raw) => decode_raw(raw, payload, name)?,
        other => {
            return Err(Error::ModelError {
                reason: format!(
                    "TCF tensor '{name}': encoding {} (0x{:04x}) has no decode path in this reader",
                    encoding_name(other),
                    other.to_u16()
                ),
            });
        }
    };

    if values.len() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "TCF tensor '{name}': decoded {} values, shape {:?} requires {expected}",
                values.len(),
                record.shape()
            ),
        });
    }
    Ok(values)
}

/// Convert a raw payload to f32, one element per stored element. Section 12.
fn decode_raw(raw: RawEncoding, payload: &[u8], name: &str) -> Result<Vec<f32>> {
    let width = usize::try_from(raw.width_bytes()).map_err(|_| Error::ModelError {
        reason: format!("TCF tensor '{name}': raw element width exceeds usize"),
    })?;
    if width == 0 || !payload.len().is_multiple_of(width) {
        return Err(Error::ModelError {
            reason: format!(
                "TCF tensor '{name}': payload of {} bytes is not a whole number of {} elements",
                payload.len(),
                encoding_name(Encoding::Raw(raw))
            ),
        });
    }

    let values = match raw {
        RawEncoding::F32 => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        RawEncoding::F16 => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        RawEncoding::Bf16 => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        RawEncoding::F8E4M3 => payload
            .iter()
            .map(|b| FP8E4M3::from_bits(*b).to_f32())
            .collect(),
        RawEncoding::F8E5M2 => payload
            .iter()
            .map(|b| FP8E5M2::from_bits(*b).to_f32())
            .collect(),
        RawEncoding::I8 => payload.iter().map(|b| f32::from(*b as i8)).collect(),
        RawEncoding::U8 => payload.iter().map(|b| f32::from(*b)).collect(),
        RawEncoding::I16 => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from(i16::from_le_bytes(*b)))
            .collect(),
        RawEncoding::U16 => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from(u16::from_le_bytes(*b)))
            .collect(),
        RawEncoding::I32 => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as f32)
            .collect(),
        RawEncoding::U32 => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b) as f32)
            .collect(),
        other => {
            return Err(Error::ModelError {
                reason: format!(
                    "TCF tensor '{name}': raw encoding 0x{:04x} has no conversion in this reader",
                    other.to_u16()
                ),
            });
        }
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures;
    use super::*;
    use tcf_core::TcfFile;

    /// Section 13.0: `x_hat_i = f32(d) * f32(q_i)`, computed here from the
    /// codes and scales the fixture chose, never from what the reader
    /// returns.
    #[test]
    fn q4_native_dequantizes_to_the_section_13_0_formula() {
        let bytes = fixtures::good_file();
        let file = TcfFile::open(&bytes).expect("fixture opens");
        let record = file.tensors()[fixtures::T_Q4];
        let payload = file.payload(&record).expect("payload");

        let values = decode_tensor_f32(&record, payload, "q4").expect("decodes");
        assert_eq!(values, fixtures::expected_q4_values());
    }

    /// CONFORMANCE.md Section 0.1: the packed bytes are checked against a
    /// packer written here, from Section 14.1, so a shared writer/reader
    /// packing bug cannot pass.
    #[test]
    fn q4_payload_bytes_match_the_section_14_1_packing_rule() {
        let bytes = fixtures::good_file();
        let file = TcfFile::open(&bytes).expect("fixture opens");
        let record = file.tensors()[fixtures::T_Q4];
        let payload = file.payload(&record).expect("payload");
        assert_eq!(payload, fixtures::expected_q4_payload().as_slice());
    }

    #[test]
    fn raw_f32_values_pass_through_unchanged() {
        let bytes = fixtures::good_file();
        let file = TcfFile::open(&bytes).expect("fixture opens");
        let record = file.tensors()[fixtures::T_RAW_F32];
        let payload = file.payload(&record).expect("payload");
        let values = decode_tensor_f32(&record, payload, "bias").expect("decodes");
        assert_eq!(values, fixtures::RAW_F32_VALUES);
    }

    /// binary16 bit patterns to their exact f32 values, written out here.
    #[test]
    fn raw_f16_converts_each_stored_bit_pattern() {
        let bytes = fixtures::good_file();
        let file = TcfFile::open(&bytes).expect("fixture opens");
        let record = file.tensors()[fixtures::T_RAW_F16];
        let payload = file.payload(&record).expect("payload");
        let values = decode_tensor_f32(&record, payload, "scale").expect("decodes");
        assert_eq!(values, vec![1.0f32, -2.0, 0.5, 0.0]);
    }

    #[test]
    fn raw_conversions_cover_every_v1_element_width() {
        assert_eq!(
            decode_raw(RawEncoding::I8, &[0xff, 0x01], "t").expect("decodes"),
            vec![-1.0f32, 1.0]
        );
        assert_eq!(
            decode_raw(RawEncoding::U8, &[0xff, 0x01], "t").expect("decodes"),
            vec![255.0f32, 1.0]
        );
        assert_eq!(
            decode_raw(RawEncoding::I16, &[0xff, 0xff], "t").expect("decodes"),
            vec![-1.0f32]
        );
        assert_eq!(
            decode_raw(RawEncoding::U16, &[0xff, 0xff], "t").expect("decodes"),
            vec![65535.0f32]
        );
        assert_eq!(
            decode_raw(RawEncoding::I32, &[0xfe, 0xff, 0xff, 0xff], "t").expect("decodes"),
            vec![-2.0f32]
        );
        assert_eq!(
            decode_raw(RawEncoding::U32, &[0x02, 0x00, 0x00, 0x00], "t").expect("decodes"),
            vec![2.0f32]
        );
        // BF16 `1.0` is the top 16 bits of f32 `1.0` (`0x3f800000`).
        assert_eq!(
            decode_raw(RawEncoding::Bf16, &[0x80, 0x3f], "t").expect("decodes"),
            vec![1.0f32]
        );
        // FP8 E4M3 `1.0`: sign 0, exponent 0111 (bias 7), mantissa 000.
        assert_eq!(
            decode_raw(RawEncoding::F8E4M3, &[0x38], "t").expect("decodes"),
            vec![1.0f32]
        );
        // FP8 E5M2 `1.0`: sign 0, exponent 01111 (bias 15), mantissa 00.
        assert_eq!(
            decode_raw(RawEncoding::F8E5M2, &[0x3c], "t").expect("decodes"),
            vec![1.0f32]
        );
    }

    #[test]
    fn a_partial_element_is_rejected_rather_than_truncated() {
        let err = decode_raw(RawEncoding::F32, &[0u8; 6], "t").expect_err("rejects");
        assert!(err.to_string().contains("whole number"), "{err}");
    }
}
