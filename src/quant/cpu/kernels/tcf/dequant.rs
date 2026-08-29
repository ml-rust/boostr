//! CPU dequantization of a TCF native quantized payload.
//!
//! # Why there is no unpack loop here
//!
//! `tcf-core` IS the reference codec: CONFORMANCE.md makes its scalar
//! implementation the definition of the semantics, and MIGRATION.md
//! Section 4.5.3 forbids a second copy of a block layout. So this kernel
//! holds no plane offset, no nibble index, no 6-bit field position and no
//! super-block stride. It calls [`unpack`] to rebuild the logical tiles, and
//! its whole job is the surrounding shape and length checks plus the
//! `Result` mapping into boostr's error type.
//!
//! That is not a placeholder. A hand-written unpack here would be a second
//! definition of the format, and Q6_K already shipped once in this codebase
//! with a wrong field order that its own reader agreed with.
//!
//! # Where the reconstruction runs
//!
//! Section 13.0's arithmetic is applied by
//! [`super::dequantize_tiles_into`], eight elements per vector iteration,
//! because `tcf-core`'s per-element `f32::from(i8)` has no shape a compiler
//! vectorizes. That is arithmetic, not format: the codes are already whole
//! bytes by then, and the two-level `(d, m)` resolution of Section 13.3 and
//! Section 13.4 still runs inside `tcf-core`. It is admitted only because
//! the gate below asserts it equals `tcf_core::dequantize` bit for bit, for
//! all seven v1 encodings — the condition this module named in advance for
//! any faster path added beside it.
//!
//! # Seam for the fused quantized matmul
//!
//! [`unpack_tiles`] is the tile-level entry point, and it decodes the WHOLE
//! tensor. The fused matmul cannot use it for that reason: it takes bounded
//! ranges from [`super::unpack_tile_range`] instead and accumulates per tile,
//! so the f32 tensor [`dequant_tcf`] builds is never materialized. Both paths
//! reach `tcf-core` through this module's shape math, so neither can disagree
//! with the other on tile count.

use tcf_core::{LogicalTile, unpack};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;

/// Rebuild the logical tiles a TCF payload encodes. Section 14.
///
/// `payload` is the tensor's logical bytes, alignment padding excluded — what
/// `TcfFile::payload` returns. The caller verifies digests first: this
/// rebuilds tiles, it does not authenticate.
///
/// # Errors
/// [`Error::ModelError`] carrying the spec's `E_*` code when the codec
/// rejects the payload — a short slice, a reserved code (Section 13.2), or a
/// scale that is NaN, infinite, or negatively signed (Section 13.1).
/// [`Error::QuantError`] when `shape` is not a tileable shape.
pub fn unpack_tiles(
    payload: &[u8],
    encoding: TcfEncoding,
    shape: &[usize],
) -> Result<Vec<LogicalTile>> {
    let tiles = encoding.tile_count(shape)?;
    unpack(payload, tiles, encoding.layout())
        .map_err(|e| tcf_error(&format!("{} unpack", encoding.name()), e))
}

/// Dequantize a TCF payload into `product(shape)` row-major f32 values.
///
/// Bit for bit what `tcf_core::unpack` followed by `tcf_core::dequantize`
/// produces, over the vectorized element loop rather than the scalar one.
///
/// # Errors
/// Every error [`unpack_tiles`] raises, plus [`Error::ModelError`] when the
/// value count disagrees with `shape` — which would mean the tile arithmetic
/// and the shape had drifted apart.
pub fn dequant_tcf(payload: &[u8], encoding: TcfEncoding, shape: &[usize]) -> Result<Vec<f32>> {
    let expected: usize = shape.iter().product();
    let tiles = unpack_tiles(payload, encoding, shape)?;
    let mut values = Vec::new();
    super::dequantize_tiles_into(&tiles, encoding.layout(), &mut values)
        .map_err(|e| tcf_error(&format!("{} dequantize", encoding.name()), e))?;

    if values.len() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{}: dequantized {} values, shape {shape:?} requires {expected}",
                encoding.name(),
                values.len(),
            ),
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::{NativeEncoding, dequantize, pack, quantize};

    /// Every v1 native quantized encoding, the two two-level forms included.
    const ENCODINGS: [NativeEncoding; 7] = [
        NativeEncoding::Q4S32T64,
        NativeEncoding::Q4AS32T64,
        NativeEncoding::Q4AS64T64,
        NativeEncoding::Q6S32T64,
        NativeEncoding::Q8S32T64,
        NativeEncoding::Q6S16DT64,
        NativeEncoding::Q4AS32DT64,
    ];

    /// `[3, 320]`: five tiles per row over three rows is 15 tiles, which is
    /// three whole 4-tile super-blocks and a partial fourth. Multi-tile,
    /// multi-super-block, and partial-trailing-super-block in one shape.
    const SHAPE: [usize; 2] = [3, 320];

    /// A deterministic input with sign changes, a flat run, and a spike, so a
    /// group's scale and minimum both move between groups.
    fn source_values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let x = i as f32;
                match i % 7 {
                    0 => 0.25,
                    3 => -(x * 0.013).sin() * 4.0,
                    _ => (x * 0.031).cos() * 1.75 - 0.4,
                }
            })
            .collect()
    }

    /// Pack a tensor with `tcf-core`'s own writer, so the bytes under test
    /// are the bytes the format defines.
    fn packed(encoding: NativeEncoding, values: &[f32]) -> Vec<u8> {
        let layout = encoding.layout();
        let dims: Vec<u64> = SHAPE.iter().map(|d| *d as u64).collect();
        let tiles = quantize(values, &dims, 2, layout).expect("quantizes");
        pack(&tiles, layout).expect("packs")
    }

    /// THE correctness gate. The kernel's output MUST equal
    /// `tcf_core::unpack` + `tcf_core::dequantize`, bit for bit, for every
    /// encoding, over a tensor with a partial trailing super-block.
    #[test]
    fn every_encoding_matches_the_tcf_core_reference_bit_for_bit() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        for native in ENCODINGS {
            let encoding = TcfEncoding::new(native);
            let payload = packed(native, &values);
            assert_eq!(
                payload.len(),
                encoding.payload_bytes(&SHAPE).expect("bytes"),
                "{} payload length",
                encoding.name()
            );

            let reference = {
                let tiles = unpack(&payload, 15, encoding.layout()).expect("unpacks");
                dequantize(&tiles, encoding.layout()).expect("dequantizes")
            };
            let got = dequant_tcf(&payload, encoding, &SHAPE).expect("dequantizes");

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let reference_bits: Vec<u32> = reference.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_bits, reference_bits, "{}", encoding.name());
            assert_eq!(got.len(), SHAPE[0] * SHAPE[1], "{}", encoding.name());
        }
    }

    /// The reconstruction is not merely self-consistent: it tracks the input
    /// within each encoding's quantization step.
    #[test]
    fn every_encoding_reconstructs_the_input_within_its_step() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        for (native, tolerance) in [
            (NativeEncoding::Q4S32T64, 0.60f32),
            (NativeEncoding::Q4AS32T64, 0.60),
            (NativeEncoding::Q4AS64T64, 0.60),
            (NativeEncoding::Q6S32T64, 0.15),
            (NativeEncoding::Q8S32T64, 0.05),
            (NativeEncoding::Q6S16DT64, 0.15),
            (NativeEncoding::Q4AS32DT64, 0.60),
        ] {
            let encoding = TcfEncoding::new(native);
            let payload = packed(native, &values);
            let got = dequant_tcf(&payload, encoding, &SHAPE).expect("dequantizes");
            let worst = values
                .iter()
                .zip(got.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= tolerance,
                "{} worst error {worst} exceeds {tolerance}",
                encoding.name()
            );
        }
    }

    /// A payload one byte short is rejected with the spec's own code, never
    /// read past its end.
    #[test]
    fn a_short_payload_is_rejected_with_the_spec_code() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q4AS32DT64);
        let mut payload = packed(NativeEncoding::Q4AS32DT64, &values);
        payload.pop();

        let err = dequant_tcf(&payload, encoding, &SHAPE).expect_err("rejects");
        assert!(err.to_string().contains("E_SECTION_BOUNDS"), "{err}");
    }

    /// The tile seam the fused matmul will use reports the same tile count
    /// the shape math does.
    #[test]
    fn the_tile_seam_yields_one_tile_per_execution_tile() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q6S16DT64);
        let payload = packed(NativeEncoding::Q6S16DT64, &values);
        let tiles = unpack_tiles(&payload, encoding, &SHAPE).expect("unpacks");
        assert_eq!(tiles.len(), 15);
        assert_eq!(
            tiles.len() as u64,
            encoding.tile_count(&SHAPE).expect("tiles")
        );
    }
}
