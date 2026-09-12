//! One tensor's payload bytes to dense row-major f32 values.
//!
//! Block encodings are GGML block streams and decode through the CPU dequant
//! kernels, via [`super::block`]. Raw encodings convert element by element
//! through numr's dtype conversions: a raw encoding stores literal values
//! with no scale of any kind.

use crate::error::{Error, Result};
use crate::tcf::{Encoding, RawEncoding, TensorRecord};
use numr::dtype::{FP8E4M3, FP8E5M2};

use super::block::decode_block_f32;
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
/// An encoding this reader cannot decode is a compile error, never a runtime
/// refusal: `Encoding` is exhaustive, so a match with no arm for it fails to
/// build.
///
/// # Errors
/// - [`Error::ModelError`] carrying the spec's `E_*` code, when the codec
///   rejects the payload.
/// - [`Error::ModelError`] when the decoded length disagrees with the shape.
pub fn decode_tensor_f32(record: &TensorRecord, payload: &[u8], name: &str) -> Result<Vec<f32>> {
    let expected = element_count(record, name)?;
    let values = match record.encoding {
        Encoding::Block(block) => decode_block_f32(block, record, payload, name)?,
        Encoding::Raw(raw) => decode_raw(raw, payload, name)?,
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
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures;
    use super::*;
    use crate::tcf::TcfFile;

    /// `d * q` per `ggml-common.h`, computed by hand in the fixture, never
    /// from what the reader returns.
    #[test]
    fn q8_0_block_dequantizes_to_the_hand_computed_values() {
        let bytes = fixtures::good_file();
        let file = TcfFile::open(&bytes).expect("fixture opens");
        let record = file.tensors()[fixtures::T_Q8];
        let payload = file.payload(&record).expect("payload");

        let values = decode_tensor_f32(&record, payload, "q8").expect("decodes");
        assert_eq!(values, fixtures::expected_q8_0_values());
        // The payload is the block stream, byte for byte.
        assert_eq!(payload, fixtures::q8_0_stream().as_slice());
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
