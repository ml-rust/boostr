//! Bounded decoding of one tile range of a TCF payload.
//!
//! # Why this exists
//!
//! [`tcf_core::unpack`] decodes a whole payload in one call and returns a
//! `Vec<LogicalTile>` covering every tile. A fused matmul cannot use that: a
//! 1.29 GB weight would become a multi-gigabyte tile vector before the first
//! dot product ran, which is the cost the fused kernel exists to remove.
//!
//! # Why a copy, and what it is NOT
//!
//! TCF stores whole planes over the whole tensor (SPECIFICATION.md Section 14):
//! all codes, then all scales, then all minima, then the per-super-block
//! values. A tile range is therefore a set of disjoint byte runs, one per
//! plane, never a single slice. [`unpack_tile_range`] gathers those runs into a
//! chunk-sized buffer that is byte-for-byte the payload of a `tiles`-tile
//! tensor, then hands it to `tcf-core`'s own reader.
//!
//! So this module holds plane *extents*, which CONFORMANCE.md Section 0.1
//! names as shareable schema-derived constants, and every one of them is read
//! off [`QuantLayout`] rather than restated here. It holds no bit position, no
//! nibble index, no field order, and no scale resolution — a second copy of
//! those is what MIGRATION.md Section 4.5.3 forbids and what shipped Q6_K with
//! a wrong field order once already. The one packing fact it does encode, the
//! Section 14.2 split of a 6-bit code plane into a low-nibble sub-plane
//! followed by a high-two-bit sub-plane, is checked in two ways: the assembled
//! chunk must equal `logical_payload_bytes(tiles)`, and the tests compare every
//! range against `tcf_core::unpack` over the whole payload.

use tcf_core::{LogicalTile, QuantLayout, unpack};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;

/// One contiguous plane region of a payload, and the unit it is charged in.
///
/// A zero stride is a plane this encoding does not spend, and contributes
/// nothing at every tile count — so the region list below is one fixed shape
/// for all seven encodings rather than a per-encoding branch.
#[derive(Debug, Clone, Copy)]
enum Region {
    /// `stride` bytes per execution tile.
    PerTile(usize),
    /// `stride` bytes per super-block, a partial trailing block charged whole.
    PerBlock(usize),
}

/// Planes of a TCF payload, in the storage order of Section 14: codes,
/// scales, minima, super-scales, super-minima.
///
/// The scale plane is charged per tile under a flat or two-level-`u8` form and
/// per super-block under the two-level 6-bit form, exactly as
/// [`QuantLayout::scale_plane_bytes`] sums the two terms; the minimum plane
/// mirrors it. A 6-bit code plane occupies two sub-planes (Section 14.2), so
/// the code entry is a pair.
fn regions(layout: QuantLayout) -> Result<[Region; 8]> {
    let geometry = layout.geometry;
    let tile = usize::from(geometry.tile);
    let code_bytes =
        usize::try_from(geometry.code_bytes_per_tile()).map_err(|_| Error::QuantError {
            reason: "TCF code stride exceeds usize".into(),
        })?;

    // Section 14.1 / 14.3 pack a tile's codes as one run; Section 14.2 splits
    // a 6-bit plane into a whole low-nibble sub-plane and a whole high-two-bit
    // sub-plane, each strided per tile.
    let (code_low, code_high) = match geometry.bits {
        4 | 8 => (code_bytes, 0),
        6 => (tile / 2, tile / 4),
        other => {
            return Err(Error::QuantError {
                reason: format!("TCF: Section 14 defines no {other}-bit code packing"),
            });
        }
    };

    let widen_u32 = |value: u32| -> Result<usize> {
        usize::try_from(value).map_err(|_| Error::QuantError {
            reason: "TCF plane stride exceeds usize".into(),
        })
    };

    Ok([
        Region::PerTile(code_low),
        Region::PerTile(code_high),
        Region::PerTile(widen_u32(layout.scale_bytes_per_tile())?),
        Region::PerBlock(widen_u32(layout.sub_scale_bytes_per_block())?),
        Region::PerTile(widen_u32(layout.min_bytes_per_tile())?),
        Region::PerBlock(widen_u32(layout.sub_min_bytes_per_block())?),
        Region::PerBlock(widen_u32(layout.super_scale_bytes_per_block())?),
        Region::PerBlock(widen_u32(layout.super_min_bytes_per_block())?),
    ])
}

/// Super-blocks covering `tiles`, a partial trailing block counted whole.
fn blocks(layout: QuantLayout, tiles: usize) -> Result<usize> {
    let per_block = usize::from(layout.tiles_per_super_block());
    if per_block == 0 {
        return Err(Error::QuantError {
            reason: "TCF layout reports a zero-tile super-block".into(),
        });
    }
    Ok(tiles.div_ceil(per_block))
}

/// Bytes `region` occupies over `tiles` tiles.
fn region_bytes(region: Region, layout: QuantLayout, tiles: usize) -> Result<usize> {
    let (unit, stride) = match region {
        Region::PerTile(stride) => (tiles, stride),
        Region::PerBlock(stride) => (blocks(layout, tiles)?, stride),
    };
    unit.checked_mul(stride).ok_or_else(|| Error::QuantError {
        reason: "TCF plane span overflows usize".into(),
    })
}

/// The offset of a range's first unit inside `region`, in units.
fn region_offset(region: Region, layout: QuantLayout, first_tile: usize) -> Result<usize> {
    match region {
        Region::PerTile(_) => Ok(first_tile),
        Region::PerBlock(_) => {
            let per_block = usize::from(layout.tiles_per_super_block());
            first_tile
                .checked_div(per_block)
                .ok_or_else(|| Error::QuantError {
                    reason: "TCF layout reports a zero-tile super-block".into(),
                })
        }
    }
}

/// A checked `a + b`, named so the error says which sum overflowed.
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| Error::QuantError {
        reason: "TCF plane offset overflows usize".into(),
    })
}

/// Decode tiles `first_tile..first_tile + tiles` of `payload`.
///
/// `scratch` is the caller's reusable gather buffer; it is cleared and refilled
/// on every call, so one buffer per worker serves a whole matmul and the kernel
/// allocates no byte buffer per range.
///
/// `first_tile` MUST be a multiple of the layout's super-block width. A
/// super-block's scales are addressed by the tile's position within its own
/// block (Section 14.6), so a range starting mid-block would read every
/// two-level group's parameters from the wrong slot — plausible numbers, not an
/// error. The end need not be aligned: a partial trailing block is charged
/// whole on both sides.
///
/// # Errors
/// [`Error::QuantError`] when `first_tile` is not block-aligned, when the range
/// runs past `total_tiles`, when `payload` is shorter than `total_tiles`
/// requires, or when the assembled chunk disagrees with the layout's own
/// payload size. [`Error::ModelError`] carrying the spec's `E_*` code when
/// `tcf-core` rejects the chunk.
pub fn unpack_tile_range(
    payload: &[u8],
    encoding: TcfEncoding,
    total_tiles: u64,
    first_tile: u64,
    tiles: u64,
    scratch: &mut Vec<u8>,
) -> Result<Vec<LogicalTile>> {
    let layout = encoding.layout();
    let name = encoding.name();

    let widen = |value: u64| -> Result<usize> {
        usize::try_from(value).map_err(|_| Error::QuantError {
            reason: format!("{name}: tile index {value} exceeds usize"),
        })
    };
    let total = widen(total_tiles)?;
    let first = widen(first_tile)?;
    let count = widen(tiles)?;

    let per_block = usize::from(layout.tiles_per_super_block());
    if per_block == 0 || !first.is_multiple_of(per_block) {
        return Err(Error::QuantError {
            reason: format!(
                "{name}: tile range must start on a {per_block}-tile super-block, got {first}"
            ),
        });
    }
    let end = add(first, count)?;
    if end > total {
        return Err(Error::QuantError {
            reason: format!("{name}: tile range {first}..{end} exceeds {total} tiles"),
        });
    }

    let expected_total = encoding_payload_bytes(layout, total_tiles, &name)?;
    if payload.len() < expected_total {
        return Err(Error::QuantError {
            reason: format!(
                "{name}: payload of {} bytes is shorter than the {expected_total} bytes {total} tiles require",
                payload.len(),
            ),
        });
    }

    let plan = regions(layout)?;
    scratch.clear();
    let mut plane_base = 0usize;
    for region in plan {
        let plane_bytes = region_bytes(region, layout, total)?;
        let offset = region_offset(region, layout, first)?;
        let stride = match region {
            Region::PerTile(stride) | Region::PerBlock(stride) => stride,
        };
        let skip = offset
            .checked_mul(stride)
            .ok_or_else(|| Error::QuantError {
                reason: format!("{name}: plane offset overflows usize"),
            })?;
        let start = add(plane_base, skip)?;
        let len = region_bytes(region, layout, count)?;
        let run = payload
            .get(start..add(start, len)?)
            .ok_or_else(|| Error::QuantError {
                reason: format!("{name}: tile range reads past the payload"),
            })?;
        scratch.extend_from_slice(run);
        plane_base = add(plane_base, plane_bytes)?;
    }

    // Two checks in one line each. The first says the region list reproduces
    // the layout's own total, so no plane was dropped or double-counted; the
    // second says the gathered chunk is exactly a `count`-tile payload, which
    // is what `unpack` is about to assume.
    if plane_base != expected_total {
        return Err(Error::QuantError {
            reason: format!(
                "{name}: planes sum to {plane_base} bytes, layout requires {expected_total}"
            ),
        });
    }
    let expected_chunk = encoding_payload_bytes(layout, tiles, &name)?;
    if scratch.len() != expected_chunk {
        return Err(Error::QuantError {
            reason: format!(
                "{name}: gathered {} bytes for {count} tiles, layout requires {expected_chunk}",
                scratch.len(),
            ),
        });
    }

    unpack(scratch, tiles, layout).map_err(|e| tcf_error(&format!("{name} unpack range"), e))
}

/// `logical_payload_bytes` narrowed to `usize`, with the encoding named.
fn encoding_payload_bytes(layout: QuantLayout, tiles: u64, name: &str) -> Result<usize> {
    let bytes = layout
        .logical_payload_bytes(tiles)
        .map_err(|e| tcf_error(&format!("{name} payload bytes"), e))?;
    usize::try_from(bytes).map_err(|_| Error::QuantError {
        reason: format!("{name}: payload of {bytes} bytes exceeds usize"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::{NativeEncoding, pack, quantize};

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
    /// the gather may not change a single field.
    #[test]
    fn every_range_matches_the_whole_payload_decode() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        for native in ENCODINGS {
            let encoding = TcfEncoding::new(native);
            let payload = packed(native, &values);
            let whole = unpack(&payload, TILES, encoding.layout()).expect("unpacks");

            let per_block = u64::from(encoding.layout().tiles_per_super_block());
            let mut scratch = Vec::new();
            for start in (0..TILES).step_by(per_block as usize) {
                for len in [per_block, per_block * 3, per_block * 7 + 1] {
                    let len = len.min(TILES - start);
                    let got =
                        unpack_tile_range(&payload, encoding, TILES, start, len, &mut scratch)
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

    /// The gather buffer never grows with the tensor: it holds one range.
    #[test]
    fn the_gather_buffer_is_sized_by_the_range_not_the_tensor() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q4AS32DT64);
        let payload = packed(NativeEncoding::Q4AS32DT64, &values);
        let mut scratch = Vec::new();
        unpack_tile_range(&payload, encoding, TILES, 0, 4, &mut scratch).expect("unpacks");
        assert_eq!(scratch.len(), 4 * 32 + 16);
        assert!(scratch.len() * 8 < payload.len());
    }

    /// A range starting mid-super-block would read two-level sub-scales from
    /// the wrong slot, so it is refused rather than decoded.
    #[test]
    fn a_range_starting_mid_super_block_is_refused() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q6S16DT64);
        let payload = packed(NativeEncoding::Q6S16DT64, &values);
        let mut scratch = Vec::new();
        let err =
            unpack_tile_range(&payload, encoding, TILES, 5, 4, &mut scratch).expect_err("refuses");
        assert!(err.to_string().contains("super-block"), "{err}");
    }

    #[test]
    fn a_range_past_the_tile_count_is_refused() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q8S32T64);
        let payload = packed(NativeEncoding::Q8S32T64, &values);
        let mut scratch = Vec::new();
        assert!(unpack_tile_range(&payload, encoding, TILES, 116, 8, &mut scratch).is_err());
    }

    #[test]
    fn a_short_payload_is_refused_before_any_gather() {
        let values = source_values(SHAPE[0] * SHAPE[1]);
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        let mut payload = packed(NativeEncoding::Q4S32T64, &values);
        payload.truncate(payload.len() - 1);
        let mut scratch = Vec::new();
        let err =
            unpack_tile_range(&payload, encoding, TILES, 0, 4, &mut scratch).expect_err("refuses");
        assert!(err.to_string().contains("shorter"), "{err}");
    }

    /// The region list must reproduce the layout's own payload size for every
    /// encoding, at a tile count with a partial trailing super-block.
    #[test]
    fn the_region_list_reproduces_every_layouts_payload_size() {
        for native in ENCODINGS {
            let layout = native.layout();
            for tiles in [1usize, 4, 15, 120] {
                let sum: usize = regions(layout)
                    .expect("regions")
                    .into_iter()
                    .map(|r| region_bytes(r, layout, tiles).expect("bytes"))
                    .sum();
                let want = layout
                    .logical_payload_bytes(tiles as u64)
                    .expect("payload bytes");
                assert_eq!(sum as u64, want, "{native:?} at {tiles} tiles");
            }
        }
    }
}
