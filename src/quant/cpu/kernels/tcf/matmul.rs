//! Fused CPU matmul against a TCF native quantized weight.
//!
//! # What "fused" means here
//!
//! The weight is never materialized as f32. The kernel walks the weight one
//! bounded tile range at a time, reconstructs that range's 64-element tiles
//! into a fixed scratch buffer, accumulates their contribution to the output,
//! and drops them. Peak working set per worker is
//! [`FUSED_TILE_CHUNK`] tiles, independent of `n` and of the weight's size — so
//! a TCF model runs at its quantized memory cost instead of the roughly
//! sevenfold f32 expansion a dequantize-then-dense-matmul path pays.
//!
//! # Allocation is fixed, not per chunk
//!
//! Every buffer this kernel needs is allocated once and reused. The output
//! slabs are one staging buffer the workers split with
//! [`rayon::slice::ParallelSliceMut::par_chunks_mut`], and each worker holds one
//! tile buffer and one reconstruction buffer that
//! [`unpack_tile_range`] and [`dequantize_into`] refill in place. So the
//! allocation count is a small constant, not a multiple of the chunk count.
//!
//! # What the plane layout costs
//!
//! GGUF packs a block's codes and its scale adjacent, so a GGUF kernel reads
//! one contiguous run per block. TCF packs whole planes over the whole tensor
//! (SPECIFICATION.md Section 14), so decoding a tile range touches up to five
//! separated read streams instead of one. `tcf-core` indexes them directly out
//! of the whole payload, so the cost is the extra streams, never a copy.
//! CONFORMANCE.md Section 8.1 leaves code-plane ordering unfrozen and settles
//! it by benchmark; this kernel is where that measurement is taken.
//!
//! # Scalar, and shaped for SIMD
//!
//! [`tile_dot`] is the single arithmetic entry point, and reconstruction runs
//! through `tcf-core`'s own [`dequantize_into`]. A SIMD unit replaces the body
//! of `tile_dot` and adds a widened reconstruction behind the same call, with
//! no change to the traversal above it.

use rayon::prelude::*;
use tcf_core::{LogicalTile, dequantize_into};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;

use super::stream::unpack_tile_range;

/// Execution tiles one worker decodes at a time.
///
/// A multiple of the four-tile super-block, so every range this kernel asks
/// for starts on a block boundary. At the v1 tile width of 64 elements this is
/// 4096 f32 of reconstruction scratch per worker — a fixed cost, whatever the
/// weight's size.
pub const FUSED_TILE_CHUNK: usize = 64;

/// `activation [M, K] × weight [N, K]^T -> output [M, N]`, against a TCF
/// native quantized weight held in its packed plane-major form.
///
/// `payload` is the tensor's logical bytes for shape `[n, k]` — the same bytes
/// the whole-tensor dequantization path takes, which this one never calls. The
/// caller verifies digests; this multiplies.
///
/// The result equals `matmul(activation, dequantize(weight))` in f32 up to
/// summation order, because both sides reconstruct each weight element through
/// the same `tcf-core` call.
///
/// # Errors
/// [`Error::QuantError`] on a shape or slice-length disagreement.
/// [`Error::ModelError`] carrying the spec's `E_*` code when `tcf-core` rejects
/// the payload.
pub fn tcf_matmul_f32(
    act: &[f32],
    payload: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    encoding: TcfEncoding,
) -> Result<()> {
    if act.len() != m.saturating_mul(k) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: activation holds {} values, {m}x{k} required",
                encoding.name(),
                act.len()
            ),
        });
    }
    if output.len() != m.saturating_mul(n) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: output holds {} values, {m}x{n} required",
                encoding.name(),
                output.len()
            ),
        });
    }
    if m == 0 || n == 0 {
        return Ok(());
    }

    let tile = encoding.tile();
    if tile == 0 || k == 0 || !k.is_multiple_of(tile) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: K={k} is not a positive multiple of the tile width {tile}",
                encoding.name()
            ),
        });
    }
    let tiles_per_row = k / tile;
    let layout = encoding.layout();
    let per_block = usize::from(layout.tiles_per_super_block());
    if per_block == 0 || !FUSED_TILE_CHUNK.is_multiple_of(per_block) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: super-block width {per_block} does not divide the chunk",
                encoding.name()
            ),
        });
    }
    let total_tiles = encoding.tile_count(&[n, k])?;

    // A range must start on a super-block boundary, and row `j` starts at tile
    // `j * tiles_per_row`. The smallest row step that lands on a boundary is
    // therefore `per_block / gcd(tiles_per_row, per_block)` — 1 whenever a row
    // is a whole number of super-blocks, which is the common case.
    let rows_per_group = per_block / gcd(tiles_per_row, per_block);

    // One staging buffer for every worker's slab, split by `par_chunks_mut`.
    // `m` and `n` are non-zero here, so the stride is non-zero, and the chunk
    // count is exactly `n.div_ceil(rows_per_group)` — one per row group, the
    // last holding a partial group's rows.
    let slab_stride = m
        .checked_mul(rows_per_group)
        .ok_or_else(|| overflow(encoding))?;
    let mut staging = vec![0.0f32; m.checked_mul(n).ok_or_else(|| overflow(encoding))?];
    staging
        .par_chunks_mut(slab_stride)
        .enumerate()
        .try_for_each_init(
            || (Vec::new(), Vec::new()),
            |(tiles, values), (group, slab)| {
                row_group(
                    act,
                    payload,
                    RowGroup {
                        group,
                        rows_per_group,
                        m,
                        k,
                        n,
                        tile,
                        tiles_per_row,
                        total_tiles,
                    },
                    encoding,
                    slab,
                    tiles,
                    values,
                )
            },
        )?;

    for (group, values) in staging.chunks(slab_stride).enumerate() {
        let row_start = group.saturating_mul(rows_per_group);
        let rows = rows_per_group.min(n.saturating_sub(row_start));
        for row in 0..m {
            for column in 0..rows {
                let value = values
                    .get(row * rows + column)
                    .copied()
                    .ok_or_else(|| short(encoding))?;
                *output
                    .get_mut(row * n + row_start + column)
                    .ok_or_else(|| short(encoding))? = value;
            }
        }
    }
    Ok(())
}

/// Which weight rows one worker owns, and the geometry it needs to place them.
#[derive(Debug, Clone, Copy)]
struct RowGroup {
    group: usize,
    rows_per_group: usize,
    m: usize,
    k: usize,
    n: usize,
    tile: usize,
    tiles_per_row: usize,
    total_tiles: u64,
}

/// Accumulate one worker's `[m, rows]` slab of the output.
///
/// The slab is row-major in the activation's row index, so a worker writes a
/// dense buffer and the caller scatters it into the strided output once. That
/// keeps every write in this function local and needs no shared mutable state.
///
/// `slab` arrives zeroed and is accumulated into. `tiles` and `values` are the
/// worker's reusable decode buffers: both are refilled per chunk, so a whole
/// row group allocates nothing.
fn row_group(
    act: &[f32],
    payload: &[u8],
    at: RowGroup,
    encoding: TcfEncoding,
    slab: &mut [f32],
    tiles: &mut Vec<LogicalTile>,
    values: &mut Vec<f32>,
) -> Result<()> {
    let layout = encoding.layout();
    let row_start = at.group.saturating_mul(at.rows_per_group);
    let rows = at.rows_per_group.min(at.n.saturating_sub(row_start));
    if rows == 0 {
        return Ok(());
    }

    let first_tile = row_start
        .checked_mul(at.tiles_per_row)
        .ok_or_else(|| overflow(encoding))?;
    let group_tiles = rows
        .checked_mul(at.tiles_per_row)
        .ok_or_else(|| overflow(encoding))?;

    let mut decoded = 0usize;
    while decoded < group_tiles {
        let take = FUSED_TILE_CHUNK.min(group_tiles - decoded);
        let range_start = first_tile
            .checked_add(decoded)
            .ok_or_else(|| overflow(encoding))?;
        unpack_tile_range(
            payload,
            encoding,
            at.total_tiles,
            u64::try_from(range_start).map_err(|_| overflow(encoding))?,
            u64::try_from(take).map_err(|_| overflow(encoding))?,
            tiles,
        )?;
        dequantize_into(tiles.as_slice(), layout, values)
            .map_err(|e| tcf_error(&format!("{} dequantize range", encoding.name()), e))?;

        for (index, weights) in values.chunks_exact(at.tile).enumerate() {
            let global = range_start
                .checked_add(index)
                .ok_or_else(|| overflow(encoding))?;
            let weight_row = global / at.tiles_per_row;
            let column = (global % at.tiles_per_row)
                .checked_mul(at.tile)
                .ok_or_else(|| overflow(encoding))?;
            let slab_column = weight_row
                .checked_sub(row_start)
                .ok_or_else(|| short(encoding))?;

            for row in 0..at.m {
                let base = row.checked_mul(at.k).ok_or_else(|| overflow(encoding))?;
                let activations = act
                    .get(base + column..base + column + at.tile)
                    .ok_or_else(|| short(encoding))?;
                let slot = slab
                    .get_mut(row * rows + slab_column)
                    .ok_or_else(|| short(encoding))?;
                *slot += tile_dot(activations, weights);
            }
        }
        decoded += take;
    }
    Ok(())
}

/// Dot product of one reconstructed tile against the activation window it
/// covers. Scalar; the SIMD unit replaces this body.
fn tile_dot(activations: &[f32], weights: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (a, w) in activations.iter().zip(weights.iter()) {
        sum += a * w;
    }
    sum
}

/// Greatest common divisor, for the row step that lands on a super-block.
fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}

/// A slice ended before the geometry said it would. The encoding is named
/// here rather than at the call site, so the happy path never builds the name.
fn short(encoding: TcfEncoding) -> Error {
    Error::QuantError {
        reason: format!("{}: fused matmul indexed past a buffer", encoding.name()),
    }
}

/// An index product exceeded `usize`.
fn overflow(encoding: TcfEncoding) -> Error {
    Error::QuantError {
        reason: format!("{}: fused matmul index overflows usize", encoding.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::{NativeEncoding, dequantize, pack, quantize, unpack};

    const ENCODINGS: [NativeEncoding; 7] = [
        NativeEncoding::Q4S32T64,
        NativeEncoding::Q4AS32T64,
        NativeEncoding::Q4AS64T64,
        NativeEncoding::Q6S32T64,
        NativeEncoding::Q8S32T64,
        NativeEncoding::Q6S16DT64,
        NativeEncoding::Q4AS32DT64,
    ];

    fn source_values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let x = (i + seed) as f32;
                match (i + seed) % 6 {
                    0 => 0.75,
                    2 => -(x * 0.011).sin() * 2.5,
                    4 => (x * 0.037).cos() * 1.5,
                    _ => (x * 0.023).sin() * 1.1 - 0.2,
                }
            })
            .collect()
    }

    fn packed(encoding: NativeEncoding, values: &[f32], shape: &[usize]) -> Vec<u8> {
        let dims: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
        let tiles = quantize(values, &dims, 2, encoding.layout()).expect("quantizes");
        pack(&tiles, encoding.layout()).expect("packs")
    }

    /// The reference side: `tcf_core::unpack` then `tcf_core::dequantize`,
    /// which CONFORMANCE.md makes the definition of the semantics, followed by
    /// a plain dense matmul.
    fn reference(
        act: &[f32],
        payload: &[u8],
        m: usize,
        k: usize,
        n: usize,
        encoding: TcfEncoding,
    ) -> Vec<f32> {
        let tiles = encoding.tile_count(&[n, k]).expect("tiles");
        let logical = unpack(payload, tiles, encoding.layout()).expect("unpacks");
        let weights = dequantize(&logical, encoding.layout()).expect("dequantizes");
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0f32;
                for index in 0..k {
                    sum += act[row * k + index] * weights[column * k + index];
                }
                out[row * n + column] = sum;
            }
        }
        out
    }

    /// THE correctness gate. Every encoding, over shapes whose tile count
    /// leaves a partial trailing super-block, and over rows that are and are
    /// not a whole number of super-blocks.
    #[test]
    fn every_encoding_matches_the_tcf_core_reference_matmul() {
        // [3, 320]: 15 tiles, five per row — a partial trailing super-block,
        // and a row step that must widen to four rows to stay block-aligned.
        // [5, 256]: 20 tiles, four per row — every row is one whole block.
        // [2, 448]: 14 tiles, seven per row — an odd row width.
        for (n, k, m) in [(3usize, 320usize, 1usize), (5, 256, 3), (2, 448, 4)] {
            for native in ENCODINGS {
                let encoding = TcfEncoding::new(native);
                let weight = source_values(n * k, 0);
                let payload = packed(native, &weight, &[n, k]);
                let act = source_values(m * k, 17);

                let mut got = vec![0.0f32; m * n];
                tcf_matmul_f32(&act, &payload, &mut got, m, k, n, encoding).expect("multiplies");
                let want = reference(&act, &payload, m, k, n, encoding);

                for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    let tolerance = 1e-3 * b.abs().max(1.0);
                    assert!(
                        (a - b).abs() <= tolerance,
                        "{} at {index}: fused {a}, reference {b}",
                        encoding.name()
                    );
                }
            }
        }
    }

    /// The fused path must never build a full f32 copy of the weight. It is
    /// asserted structurally, on the source: this module may not reach for the
    /// whole-tensor dequantization entry point at all.
    ///
    /// The needle is assembled at runtime so this assertion does not match
    /// itself in the file it reads.
    #[test]
    fn the_fused_path_never_calls_the_whole_tensor_dequant_entry_point() {
        let source = include_str!("matmul.rs");
        let entry_point = format!("dequant_{}", "tcf");
        let whole_tensor_seam = format!("unpack_{}(", "tiles");
        assert!(
            !source.contains(&entry_point),
            "{entry_point} is reachable here"
        );
        assert!(
            !source.contains(&whole_tensor_seam),
            "whole-tensor unpack is reachable here"
        );
    }

    /// The working set is fixed by the chunk, not by the weight: a worker holds
    /// 64 tiles of reconstruction whether the weight has 15 tiles or millions.
    #[test]
    fn the_working_set_is_bounded_by_the_chunk_not_the_weight() {
        let tile = TcfEncoding::new(NativeEncoding::Q4S32T64).tile();
        let scratch = FUSED_TILE_CHUNK * tile;
        assert_eq!(scratch, 4096);
        // A 4096x4096 weight is four thousand times this working set.
        assert!(scratch * 4000 < 4096 * 4096);
        assert!(FUSED_TILE_CHUNK.is_multiple_of(4));
    }

    /// A weight the payload cannot back is refused, not read past.
    #[test]
    fn a_short_payload_is_refused() {
        let encoding = TcfEncoding::new(NativeEncoding::Q6S16DT64);
        let weight = source_values(3 * 320, 0);
        let mut payload = packed(NativeEncoding::Q6S16DT64, &weight, &[3, 320]);
        payload.truncate(payload.len() - 1);
        let act = source_values(320, 5);
        let mut out = vec![0.0f32; 3];
        assert!(tcf_matmul_f32(&act, &payload, &mut out, 1, 320, 3, encoding).is_err());
    }

    #[test]
    fn a_k_that_is_not_a_whole_tile_is_refused() {
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        let mut out = vec![0.0f32; 2];
        let act = vec![0.0f32; 100];
        assert!(tcf_matmul_f32(&act, &[], &mut out, 1, 100, 2, encoding).is_err());
    }

    #[test]
    fn the_row_step_lands_every_range_on_a_super_block() {
        // Five tiles per row against a four-tile block needs four rows.
        assert_eq!(4 / gcd(5, 4), 4);
        // Four tiles per row needs one.
        assert_eq!(4 / gcd(4, 4), 1);
        // Two tiles per row needs two.
        assert_eq!(4 / gcd(2, 4), 2);
        // A flat layout has one tile per block, so every row starts a range.
        assert_eq!(1 / gcd(5, 1), 1);
    }
}
