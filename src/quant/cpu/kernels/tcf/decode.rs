//! Tile and group walk over a TCF logical tile run, into f32.
//!
//! # Why this exists beside `tcf-core`'s own reconstruction
//!
//! [`tcf_core::dequantize_into`] is the reference: CONFORMANCE.md makes its
//! scalar implementation the definition of the semantics. Its element loop
//! is one `i8` load, one `f32::from`, one multiply and one store per value,
//! which is a shape no compiler vectorizes on its own — `f32::from(i8)` per
//! element never becomes a `vcvtdq2ps`.
//!
//! So this module keeps the reference's control flow verbatim — the same
//! geometry checks, the same rejection points, the same per-group
//! resolution through [`QuantLayout::group_values`] — and replaces only the
//! element loop, with [`Decoder`]. It holds no bit position, no plane
//! offset, no nibble index and no field order: by the time a
//! [`LogicalTile`] exists, `tcf-core` has already expanded every code into a
//! whole byte. So this is a second copy of Section 13.0's arithmetic, never
//! of the format, and MIGRATION.md Section 4.5.3's rule against a second
//! copy of a block layout is not reached.
//!
//! The bit-identity that permits it is asserted, not asserted-in-prose:
//! [`super::dequant_tcf`]'s test compares this path against
//! `tcf_core::unpack` plus `tcf_core::dequantize` bit for bit, for all seven
//! v1 encodings, and the test below does the same directly.

use tcf_core::{Code64, LogicalTile, QuantLayout, TcfError};

use crate::quant::cpu::kernels::simd::tcf_decode::Decoder;

/// Reconstruct the f32 values `tiles` encode into `out`, per `layout`.
///
/// Bit for bit what [`tcf_core::dequantize_into`] produces, for every
/// encoding and every input: the same expression, in the same operand
/// order, with the same rounding, applied eight lanes at a time.
///
/// `out` is cleared and refilled, so one buffer serves a whole streaming
/// matmul and a chunked decode allocates nothing per chunk. On an error
/// `out` holds the values written before the failure and MUST be treated as
/// scratch — the contract the reference has.
///
/// # Errors
/// [`TcfError::InvalidQuantGeometry`] for a tile width other than 64, a
/// group count disagreeing with the layout, a tile whose code signedness
/// disagrees with `layout.geometry.symmetric`, a tile missing a super-scale
/// or super-minimum the layout requires, or an asymmetric group carrying no
/// minimum. [`TcfError::InvalidQuantCode`] for a reserved 6-bit
/// sub-minimum, raised by [`QuantLayout::group_values`].
pub fn dequantize_tiles_into(
    tiles: &[LogicalTile],
    layout: QuantLayout,
    out: &mut Vec<f32>,
) -> Result<(), TcfError> {
    let tile_width = usize::from(layout.geometry.tile);
    // Sized once rather than grown per group, so the element loop writes
    // into a slice a vector store can address. The zero fill this costs is
    // one store pass over the same bytes the decode overwrites immediately.
    out.clear();
    out.resize(tiles.len().saturating_mul(tile_width), 0.0);

    let mut written = 0usize;
    let outcome = fill(tiles, layout, out.as_mut_slice(), &mut written);
    if outcome.is_err() {
        out.truncate(written);
    }
    outcome
}

/// Write every tile's reconstruction into an already-sized `out`.
///
/// `written` tracks the prefix holding real values, so a rejection midway
/// leaves the caller the same prefix the reference would have left.
fn fill(
    tiles: &[LogicalTile],
    layout: QuantLayout,
    out: &mut [f32],
    written: &mut usize,
) -> Result<(), TcfError> {
    let geometry = layout.geometry;
    let groups_per_tile = usize::from(layout.checked_groups_per_tile()?);
    let tile_width = usize::from(geometry.tile);
    let group_width = usize::from(geometry.group);
    // One feature probe per call, not per group: a 64-tile chunk resolves
    // 128 groups, and a probe is an atomic load no branch predictor removes.
    let decoder = Decoder::detect();

    for (index, tile) in tiles.iter().enumerate() {
        // Names the tile in a Section 13.4 code rejection. A tile count past
        // `u32::MAX` cannot address a payload anyway.
        let tile_index = u32::try_from(index).unwrap_or(u32::MAX);
        if tile.group_count() != groups_per_tile {
            return Err(TcfError::InvalidQuantGeometry);
        }
        // A tile carrying a super-scale under a flat layout, or lacking one
        // under a two-level layout, would reconstruct against the wrong
        // scale and return plausible wrong numbers rather than fail.
        if tile.super_scale().is_some() != layout.has_super_scale()
            || tile.super_min().is_some() != layout.has_super_min()
        {
            return Err(TcfError::InvalidQuantGeometry);
        }
        let super_scale = tile.super_scale();
        let super_min = tile.super_min();

        let mut start = 0usize;
        for group_index in 0..groups_per_tile {
            if start >= tile_width {
                break;
            }
            let end = start.saturating_add(group_width).min(tile_width);
            let params = tile
                .group(group_index)
                .ok_or(TcfError::InvalidQuantGeometry)?;
            // Section 13.3, Section 13.4: under a two-level form neither the
            // scale nor the minimum is the stored field alone. Resolved by
            // `tcf-core` itself, once per group, before this group's
            // elements are written — so a rejection blames the same group,
            // and leaves the same prefix, that the reference blames.
            let values = layout.group_values(params, super_scale, super_min, tile_index)?;

            let base = index
                .checked_mul(tile_width)
                .and_then(|b| b.checked_add(start))
                .ok_or(TcfError::InvalidQuantGeometry)?;
            let stop = base
                .checked_add(end.saturating_sub(start))
                .ok_or(TcfError::InvalidQuantGeometry)?;
            let slots = out
                .get_mut(base..stop)
                .ok_or(TcfError::InvalidQuantGeometry)?;

            match (&tile.code, geometry.symmetric) {
                (Code64::Signed(q), true) => {
                    let codes = q.get(start..end).ok_or(TcfError::InvalidQuantGeometry)?;
                    decoder.signed(codes, values.scale, slots);
                }
                (Code64::Unsigned(u), false) => {
                    let codes = u.get(start..end).ok_or(TcfError::InvalidQuantGeometry)?;
                    let min = values.min.ok_or(TcfError::InvalidQuantGeometry)?;
                    decoder.unsigned(codes, values.scale, min, slots);
                }
                _ => return Err(TcfError::InvalidQuantGeometry),
            }
            *written = stop;
            start = end;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::{NativeEncoding, dequantize, pack, quantize, unpack};

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

    /// `[3, 320]`: 15 tiles, five per row — three whole four-tile
    /// super-blocks and a partial fourth.
    const SHAPE: [usize; 2] = [3, 320];

    /// Compare by bits, never by `==`: bit-identity is the property under
    /// test, and `-0.0 == 0.0` would hide a difference.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Sign changes, a flat run, and a spike, so a group's scale and its
    /// minimum both move between groups.
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

    fn tiles_of(native: NativeEncoding, values: &[f32]) -> Vec<LogicalTile> {
        let layout = native.layout();
        let dims: Vec<u64> = SHAPE.iter().map(|d| *d as u64).collect();
        let logical = quantize(values, &dims, 2, layout).expect("quantizes");
        let payload = pack(&logical, layout).expect("packs");
        unpack(&payload, 15, layout).expect("unpacks")
    }

    /// THE correctness gate. The vectorized reconstruction MUST equal
    /// `tcf_core::dequantize` bit for bit, for every encoding, over a tensor
    /// with a partial trailing super-block.
    #[test]
    fn every_encoding_matches_the_tcf_core_reference_bit_for_bit() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        for native in ENCODINGS {
            let layout = native.layout();
            let tiles = tiles_of(native, &values);
            let reference = dequantize(&tiles, layout).expect("dequantizes");

            let mut got = Vec::new();
            dequantize_tiles_into(&tiles, layout, &mut got).expect("dequantizes");

            assert_eq!(bits(&got), bits(&reference), "{native:?}");
            assert_eq!(got.len(), SHAPE[0] * SHAPE[1], "{native:?}");
        }
    }

    /// A reused buffer is refilled, never appended to: decoding a shorter
    /// run after a longer one leaves no stale tail.
    #[test]
    fn a_reused_buffer_holds_only_the_current_run() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let layout = NativeEncoding::Q8S32T64.layout();
        let tiles = tiles_of(NativeEncoding::Q8S32T64, &values);

        let mut out = Vec::new();
        dequantize_tiles_into(&tiles, layout, &mut out).expect("dequantizes");
        assert_eq!(out.len(), 15 * 64);

        let head = tiles.get(..4).expect("four tiles");
        dequantize_tiles_into(head, layout, &mut out).expect("dequantizes");
        assert_eq!(out.len(), 4 * 64);

        let reference = dequantize(head, layout).expect("dequantizes");
        assert_eq!(bits(&out), bits(&reference));
    }

    /// A tile whose code signedness disagrees with the geometry is refused,
    /// exactly as the reference refuses it.
    #[test]
    fn a_signedness_mismatch_is_refused() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let tiles = tiles_of(NativeEncoding::Q8S32T64, &values);
        let asymmetric = NativeEncoding::Q4AS32T64.layout();
        let mut out = Vec::new();
        assert_eq!(
            dequantize_tiles_into(&tiles, asymmetric, &mut out),
            dequantize(&tiles, asymmetric).map(|_| ())
        );
        assert_eq!(
            dequantize_tiles_into(&tiles, asymmetric, &mut out),
            Err(TcfError::InvalidQuantGeometry)
        );
    }

    #[test]
    fn an_empty_tile_run_reconstructs_to_no_values() {
        let mut out = vec![1.0f32; 8];
        dequantize_tiles_into(&[], NativeEncoding::Q8S32T64.layout(), &mut out).expect("empty");
        assert!(out.is_empty());
    }
}
