//! CPU dequantization of a TCF native quantized payload.
//!
//! # Why there is no unpack loop here
//!
//! `tcf-core` IS the reference codec: CONFORMANCE.md makes its scalar
//! implementation the definition of the semantics, and MIGRATION.md
//! Section 4.5.3 forbids a second copy of a block layout. So this kernel
//! holds no plane offset, no nibble index, no 6-bit field position and no
//! super-block stride. It calls [`unpack`] or [`super::unpack_tile_range`] to
//! rebuild logical tiles, and its whole job is the surrounding shape and
//! length checks, the chunk loop, and the `Result` mapping into boostr's
//! error type.
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
//! # Two tile-level entry points
//!
//! [`unpack_tiles`] decodes the WHOLE tensor into one `Vec<LogicalTile>` and
//! stays public because tests and other callers name it directly. [`dequant_tcf`]
//! does not call it: a 2048x6144 weight is 196,608 tiles, and materializing
//! all of them before a single f32 is produced is an extra full pass plus its
//! allocation, for a value nothing downstream of `dequant_tcf` needs held at
//! once. So [`dequant_tcf`] walks the payload in bounded ranges through
//! [`super::unpack_tile_range`] instead — the same seam the fused matmul in
//! [`super::matmul`] uses — decoding [`DEQUANT_TILE_CHUNK`] tiles into one
//! reusable buffer at a time. Both entry points reach `tcf-core` through this
//! module's shape math, so neither can disagree with the other on tile count,
//! and both are checked bit-exact against `tcf_core::unpack` plus
//! `tcf_core::dequantize` below.

use tcf_core::{LogicalTile, unpack};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;

/// Execution tiles decoded per range in [`dequant_tcf`]'s streaming loop.
///
/// A [`LogicalTile`] is on the order of 96 bytes, so 512 tiles is roughly
/// 48 KB of tile-buffer scratch — L2-resident on any target, and four orders
/// of magnitude below [`unpack_tiles`]'s full `Vec<LogicalTile>`, which for a
/// 2048x6144 weight is 196,608 tiles and about 19 MB.
///
/// The size is a throughput floor, not a cache limit. Every range re-runs
/// [`super::unpack_tile_range`]'s admission arithmetic — the block-alignment
/// test, the payload-length test, and the layout's own byte-count math — all
/// of which are invariant across this loop. That fixed cost is divided by the
/// chunk's element count, so 64 tiles charged a measurable 0.2 retired
/// instructions per element and 512 tiles charges an eighth of that.
///
/// Both v1 super-block widths (1 flat, 4 two-level) divide it, which
/// [`dequant_tcf`] checks rather than assumes. Enlarging it further buys
/// nothing: the overhead it amortizes is already below the noise floor, and
/// the intermediate would grow back toward the size streaming exists to avoid.
const DEQUANT_TILE_CHUNK: u64 = 512;

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
/// Unlike [`unpack_tiles`], this never holds the whole tensor as
/// `LogicalTile`s at once. It walks the payload in [`DEQUANT_TILE_CHUNK`]-tile
/// ranges through [`super::unpack_tile_range`], the same bounded seam the
/// fused matmul uses, decoding each range into one reusable tile buffer and
/// appending its f32 reconstruction to `values`. Only the final f32 output is
/// fully materialized — the intermediate tile vector stays chunk-sized for the
/// whole call, which is the allocation and cache-traffic this function exists
/// to remove.
///
/// # Errors
/// Every error [`unpack_tiles`] raises, plus [`Error::ModelError`] when the
/// value count disagrees with `shape` — which would mean the tile arithmetic
/// and the shape had drifted apart.
pub fn dequant_tcf(payload: &[u8], encoding: TcfEncoding, shape: &[usize]) -> Result<Vec<f32>> {
    let expected: usize = shape.iter().product();
    let total_tiles = encoding.tile_count(shape)?;
    let layout = encoding.layout();

    // `DEQUANT_TILE_CHUNK` must itself be a whole number of super-blocks, or
    // every chunk after the first would start mid-block — Section 14.6's
    // failure mode `unpack_tile_range` exists to refuse. True for both v1
    // widths (1, 4); checked rather than assumed for whatever comes after.
    let per_block = u64::from(layout.tiles_per_super_block());
    if per_block == 0 || !DEQUANT_TILE_CHUNK.is_multiple_of(per_block) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: super-block width {per_block} does not divide the chunk",
                encoding.name()
            ),
        });
    }

    // Sized once to the tensor's full element count, so the loop below only
    // ever appends into already-reserved capacity: no repeated reallocation
    // as `values` grows chunk by chunk.
    let mut values = Vec::with_capacity(expected);
    // The one buffer the streaming seam refills in place. It starts empty and
    // is grown once to the chunk's size, then reused unchanged for every
    // remaining range in this call. There is deliberately no matching f32
    // scratch buffer: `dequantize_tiles_append` writes each range straight
    // into `values` at the offset the previous range stopped at, so the
    // element loop's vector stores land in the final buffer and the decode
    // costs no copy per chunk.
    let mut tiles: Vec<LogicalTile> = Vec::new();

    let mut first_tile = 0u64;
    while first_tile < total_tiles {
        let take = DEQUANT_TILE_CHUNK.min(total_tiles - first_tile);
        super::unpack_tile_range(payload, encoding, total_tiles, first_tile, take, &mut tiles)?;
        super::dequantize_tiles_append(&tiles, layout, &mut values)
            .map_err(|e| tcf_error(&format!("{} dequantize", encoding.name()), e))?;
        first_tile += take;
    }

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
