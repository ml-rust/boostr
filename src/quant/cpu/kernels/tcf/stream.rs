//! Bounded decoding of one tile range of a TCF payload.
//!
//! # Why this exists
//!
//! [`tcf_core::unpack`] decodes a whole payload in one call and returns a
//! `Vec<LogicalTile>` covering every tile. A fused matmul cannot use that: a
//! 1.29 GB weight would become a multi-gigabyte tile vector before the first
//! dot product ran, which is the cost the fused kernel exists to remove.
//!
//! # No gather, and no second copy of the plane layout
//!
//! TCF stores whole planes over the whole tensor (SPECIFICATION.md Section 14):
//! all codes, then all scales, then all minima, then the per-super-block
//! values. `tcf-core`'s [`unpack_range_into`] indexes those planes directly
//! from the WHOLE payload, so a range needs no contiguity and this module
//! copies nothing into a scratch buffer first.
//!
//! So this module holds no plane extent, no bit position, no nibble index, no
//! field order, and no scale resolution — including no copy of Section 14.2's
//! low-nibble / high-two-bit sub-plane split. A second copy of those is what
//! MIGRATION.md Section 4.5.3 forbids and what shipped Q6_K with a wrong field
//! order once already. What is left here is the range's admission rules and the
//! `Result` mapping into boostr's error type.

use tcf_core::{LogicalTile, QuantLayout, TcfError, unpack_range_into};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;

/// Decode tiles `first_tile..first_tile + tiles` of `payload` into `out`.
///
/// `payload` is the WHOLE tensor payload of `total_tiles` tiles, never a
/// chunk: Section 14's planes are addressed absolutely, and the decode indexes
/// them where they lie.
///
/// `out` is the caller's reusable tile buffer; it is cleared and refilled on
/// every call, so one buffer per worker serves a whole matmul and the kernel
/// allocates nothing per range. On an error `out` holds whatever was decoded
/// before the failure and MUST be treated as scratch.
///
/// `first_tile` MUST be a multiple of the layout's super-block width. A
/// super-block's scales are addressed by the tile's position within its own
/// block (Section 14.6), so a range starting mid-block would read every
/// two-level group's parameters from the wrong slot — plausible numbers, not an
/// error. The end need not be aligned: a range may stop inside a super-block.
///
/// # Errors
/// [`Error::QuantError`] when `first_tile` is not block-aligned, when the range
/// runs past `total_tiles`, or when `payload` is shorter than `total_tiles`
/// requires. [`Error::ModelError`] carrying the spec's `E_*` code when
/// `tcf-core` rejects the payload.
pub fn unpack_tile_range(
    payload: &[u8],
    encoding: TcfEncoding,
    total_tiles: u64,
    first_tile: u64,
    tiles: u64,
    out: &mut Vec<LogicalTile>,
) -> Result<()> {
    let layout = encoding.layout();

    let per_block = u64::from(layout.tiles_per_super_block());
    if per_block == 0 || !first_tile.is_multiple_of(per_block) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: tile range must start on a {per_block}-tile super-block, got {first_tile}",
                encoding.name()
            ),
        });
    }
    let end = first_tile
        .checked_add(tiles)
        .ok_or_else(|| Error::QuantError {
            reason: format!("{}: tile range end overflows u64", encoding.name()),
        })?;
    if end > total_tiles {
        return Err(Error::QuantError {
            reason: format!(
                "{}: tile range {first_tile}..{end} exceeds {total_tiles} tiles",
                encoding.name()
            ),
        });
    }

    let expected_total = encoding_payload_bytes(layout, total_tiles, encoding)?;
    if payload.len() < expected_total {
        // A short payload is a Section 14 bounds violation, and the spec gives
        // it a code. Raising boostr's own `QuantError` here would shadow that
        // code with a local one, so a caller that checks conformance behaviour
        // would see a short payload reported differently depending on whether
        // it entered through this bounded seam or through `tcf_core::unpack`.
        // The precheck stays — it is what keeps `out` empty rather than
        // half-decoded — but it reports what the codec itself would.
        return Err(tcf_error(
            &format!(
                "{}: payload of {} bytes is shorter than the {expected_total} bytes {total_tiles} tiles require",
                encoding.name(),
                payload.len(),
            ),
            TcfError::SectionBounds { section: "payload" },
        ));
    }

    unpack_range_into(payload, total_tiles, first_tile, tiles, layout, out)
        .map_err(|e| tcf_error(&format!("{} unpack range", encoding.name()), e))
}

/// `logical_payload_bytes` narrowed to `usize`, with the encoding named.
fn encoding_payload_bytes(layout: QuantLayout, tiles: u64, encoding: TcfEncoding) -> Result<usize> {
    let bytes = layout
        .logical_payload_bytes(tiles)
        .map_err(|e| tcf_error(&format!("{} payload bytes", encoding.name()), e))?;
    usize::try_from(bytes).map_err(|_| Error::QuantError {
        reason: format!(
            "{}: payload of {bytes} bytes exceeds usize",
            encoding.name()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::{NativeEncoding, pack, quantize, unpack};

    const ENCODINGS: [NativeEncoding; 7] = [
        NativeEncoding::Q4S32T64,
        NativeEncoding::Q4AS32T64,
        NativeEncoding::Q4AS64T64,
        NativeEncoding::Q6S32T64,
        NativeEncoding::Q8S32T64,
        NativeEncoding::Q6S16DT64,
        NativeEncoding::Q4AS32DT64,
    ];

    /// 24 rows of 320 columns is 120 tiles: 30 whole super-blocks, and every
    /// row is five tiles, which is not a multiple of the four-tile block.
    const SHAPE: [usize; 2] = [24, 320];
    const TILES: u64 = 120;

    fn source_values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let x = i as f32;
                match i % 5 {
                    0 => 0.5,
                    2 => -(x * 0.017).sin() * 3.0,
                    _ => (x * 0.029).cos() * 1.25 - 0.3,
                }
            })
            .collect()
    }

    fn packed(encoding: NativeEncoding, values: &[f32]) -> Vec<u8> {
        let dims: Vec<u64> = SHAPE.iter().map(|d| *d as u64).collect();
        let tiles = quantize(values, &dims, 2, encoding.layout()).expect("quantizes");
        pack(&tiles, encoding.layout()).expect("packs")
    }

    /// Every range this seam decodes MUST equal the same tiles of a whole
    /// payload decoded by `tcf-core` in one call. That is the whole contract:
    /// the range may not change a single field.
    #[test]
    fn every_range_matches_the_whole_payload_decode() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        for native in ENCODINGS {
            let encoding = TcfEncoding::new(native);
            let payload = packed(native, &values);
            let whole = unpack(&payload, TILES, encoding.layout()).expect("unpacks");

            let per_block = u64::from(encoding.layout().tiles_per_super_block());
            let mut got = Vec::new();
            for start in (0..TILES).step_by(per_block as usize) {
                for len in [per_block, per_block * 3, per_block * 7 + 1] {
                    let len = len.min(TILES - start);
                    unpack_tile_range(&payload, encoding, TILES, start, len, &mut got)
                        .expect("unpacks range");
                    let want = whole
                        .get(start as usize..(start + len) as usize)
                        .expect("in range");
                    assert_eq!(
                        got.as_slice(),
                        want,
                        "{} at {start}..{len}",
                        encoding.name()
                    );
                }
            }
        }
    }

    /// The decode buffer holds one range, never the tensor, and a second range
    /// of the same width reuses it without reallocating.
    #[test]
    fn the_decode_buffer_is_sized_by_the_range_and_reused() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q4AS32DT64);
        let payload = packed(NativeEncoding::Q4AS32DT64, &values);
        let mut out = Vec::new();

        unpack_tile_range(&payload, encoding, TILES, 0, 4, &mut out).expect("unpacks");
        assert_eq!(out.len(), 4);
        assert!((out.len() as u64) * 8 < TILES);
        let capacity = out.capacity();

        unpack_tile_range(&payload, encoding, TILES, 4, 4, &mut out).expect("unpacks");
        assert_eq!(out.len(), 4);
        assert_eq!(out.capacity(), capacity, "a second range reuses the buffer");
    }

    /// The property the old gather needed a size check for: this module owns no
    /// plane-extent arithmetic at all, so there is nothing here to disagree with
    /// Section 14. Asserted on the source, with every needle assembled at
    /// runtime so the assertion does not match itself.
    #[test]
    fn the_range_seam_holds_no_plane_extent_arithmetic_of_its_own() {
        let source = include_str!("stream.rs");
        for needle in [
            format!("code_bytes_per_{}", "tile"),
            format!("scale_bytes_per_{}", "tile"),
            format!("min_bytes_per_{}", "tile"),
            format!("sub_scale_bytes_per_{}", "block"),
            format!("super_scale_bytes_per_{}", "block"),
            format!("extend_from_{}", "slice"),
        ] {
            assert!(!source.contains(&needle), "{needle} is reachable here");
        }
    }

    /// A range starting mid-super-block would read two-level sub-scales from
    /// the wrong slot, so it is refused rather than decoded.
    #[test]
    fn a_range_starting_mid_super_block_is_refused() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q6S16DT64);
        let payload = packed(NativeEncoding::Q6S16DT64, &values);
        let mut out = Vec::new();
        let err =
            unpack_tile_range(&payload, encoding, TILES, 5, 4, &mut out).expect_err("refuses");
        assert!(err.to_string().contains("super-block"), "{err}");
    }

    #[test]
    fn a_range_past_the_tile_count_is_refused() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q8S32T64);
        let payload = packed(NativeEncoding::Q8S32T64, &values);
        let mut out = Vec::new();
        assert!(unpack_tile_range(&payload, encoding, TILES, 116, 8, &mut out).is_err());
    }

    #[test]
    fn a_short_payload_is_refused_before_any_decode() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        let mut payload = packed(NativeEncoding::Q4S32T64, &values);
        payload.truncate(payload.len() - 1);
        let mut out = Vec::new();
        let err =
            unpack_tile_range(&payload, encoding, TILES, 0, 4, &mut out).expect_err("refuses");
        assert!(err.to_string().contains("shorter"), "{err}");
        assert!(out.is_empty(), "no tile is decoded from a short payload");
    }
}
