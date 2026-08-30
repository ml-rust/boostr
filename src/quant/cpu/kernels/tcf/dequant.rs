//! CPU dequantization of a TCF native quantized payload.
//!
//! # Why there is no unpack loop here
//!
//! `tcf-core` IS the reference codec: CONFORMANCE.md makes its scalar
//! implementation the definition of the semantics, and MIGRATION.md
//! Section 4.5.3 forbids a second copy of a block layout. So this kernel
//! holds no plane offset, no nibble index, no 6-bit field position and no
//! super-block stride. It calls [`unpack`] or [`for_each_group`] to reach the
//! payload's contents, and its whole job is the surrounding shape and length
//! checks, the element loop, and the `Result` mapping into boostr's error
//! type.
//!
//! That is not a placeholder. A hand-written unpack here would be a second
//! definition of the format, and Q6_K already shipped once in this codebase
//! with a wrong field order that its own reader agreed with.
//!
//! # Where the reconstruction runs
//!
//! Section 13.0's arithmetic is applied by [`Decoder`], eight elements per
//! vector iteration, because `tcf-core`'s per-element `f32::from(i8)` has no
//! shape a compiler vectorizes. That is arithmetic, not format: the codes
//! [`for_each_group`] hands over are already whole bytes, and the two-level
//! `(d, m)` resolution of Section 13.3 and Section 13.4 has already run
//! inside `tcf-core`. It is admitted only because the gate below asserts it
//! equals `tcf_core::dequantize` bit for bit, for all seven v1 encodings —
//! the condition this module named in advance for any faster path added
//! beside it.
//!
//! # Two entry points, and why only one of them builds tiles
//!
//! [`unpack_tiles`] decodes the WHOLE tensor into one `Vec<LogicalTile>` and
//! stays public because tests and other callers name it directly.
//! [`dequant_tcf`] builds no [`LogicalTile`] at all: a 2048x6144 weight is
//! 196,608 tiles, and routing every element through that intermediate cost
//! 6.34 of 12.50 retired instructions per element — half the work, for a
//! value nothing downstream holds. So it walks the payload through
//! `tcf-core`'s [`for_each_group`], which streams the planes itself and
//! yields one group's widened codes and its resolved `(d, m)` at a time,
//! straight into the vector element loop. Both entry points reach `tcf-core`
//! through this module's shape math, so neither can disagree with the other
//! on tile count, and both are checked bit-exact against `tcf_core::unpack`
//! plus `tcf_core::dequantize` below.

use tcf_core::{GroupCodes, LogicalTile, TcfError, for_each_group, unpack};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;
use crate::quant::cpu::kernels::simd::tcf_decode::Decoder;

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
/// Unlike [`unpack_tiles`], this materializes no intermediate at all. One
/// [`for_each_group`] call covers the whole tensor: `tcf-core` walks the
/// planes, resolves each group's `(d, m)` once, and hands over that group's
/// widened codes, and the closure below reconstructs them directly into
/// `values`. Groups arrive in tile order, so appending is already row-major
/// and no gather or reorder is needed. Only the f32 output is ever fully
/// held, which is the allocation and the cache traffic this function exists
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

    // Sized once to the tensor's full element count, so the walk below only
    // ever grows into already-reserved capacity: no reallocation as `values`
    // fills group by group. It is a capacity, not a length, because a shape
    // whose tile arithmetic yields MORE elements than `expected` must still
    // reach the count check at the end rather than be refused mid-decode.
    // Sized and zeroed ONCE, not grown per group. A tensor resolves hundreds
    // of thousands of groups, and growing per group charged each one a capacity
    // check plus a 128-byte zero fill the decoder overwrote immediately. One
    // `vec![0.0; expected]` is a single bulk store pass the allocator and libc
    // already do at full width, after which every group writes into a slice
    // that is simply there.
    let mut values = vec![0.0f32; expected];
    // How many elements the walk has written. `values` is at full length from
    // the start, so its own length can no longer detect tile arithmetic that
    // disagrees with `shape` — this cursor is what the count check below reads.
    let mut written = 0usize;
    // One feature probe per call, not per group: a tensor resolves hundreds
    // of thousands of groups, and a probe is an atomic load that no branch
    // predictor removes.
    let decoder = Decoder::detect();

    for_each_group(
        payload,
        total_tiles,
        0,
        total_tiles,
        layout,
        |_tile, codes, group| {
            // The group's slot in the already-sized output. `get_mut` is also
            // the over-count guard: a walk that yielded more elements than
            // `shape` admits runs off the end and is refused here rather than
            // growing the buffer past what the caller asked for.
            let start = written;
            let end = start
                .checked_add(codes.len())
                .ok_or(TcfError::InvalidQuantGeometry)?;
            let slots = values
                .get_mut(start..end)
                .ok_or(TcfError::InvalidQuantGeometry)?;
            // `group.scale` and `group.min` are already resolved: Section 13.3's
            // and Section 13.4's two-level products ran inside `tcf-core`, so
            // what is left here is Section 13.0's `d * f32(q)` and Section
            // 13.0.1's `d * f32(u) + m` and nothing else. The minimum is the
            // asymmetric form's second operand, so it is required exactly where
            // the codes are unsigned, and absent everywhere else.
            match codes {
                GroupCodes::Signed(q) => decoder.signed(q, group.scale, slots),
                GroupCodes::Unsigned(u) => {
                    let min = group.min.ok_or(TcfError::InvalidQuantGeometry)?;
                    decoder.unsigned(u, group.scale, min, slots);
                }
            }
            written = end;
            Ok(())
        },
    )
    .map_err(|e| tcf_error(&format!("{} dequantize", encoding.name()), e))?;

    if written != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{}: dequantized {} values, shape {shape:?} requires {expected}",
                encoding.name(),
                written,
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
