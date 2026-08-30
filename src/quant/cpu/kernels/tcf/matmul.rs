//! Fused CPU matmul against a TCF native quantized weight.
//!
//! # What "fused" means here
//!
//! The weight is never materialized as f32. A worker walks its own rows
//! through `tcf-core`'s [`for_each_group`], reconstructs one execution tile
//! into a stack buffer group by group, accumulates that whole tile against
//! every activation row, and overwrites the buffer with the next tile. The
//! working set below the output is one tile — 64 f32 at the v1 tile width —
//! whatever `m`, `n`, and the weight's size are.
//!
//! # Why the decoded weights never reach memory
//!
//! An earlier shape decoded a fixed tile range into an f32 slab and walked
//! that slab tile by tile. At `m = 1` every decoded weight was stored and
//! reloaded for exactly one dot, costing +8.1 retired instructions per
//! multiply-accumulate over this kernel's own dequantization — where a GGUF
//! fused gemv costs +0.63 over its. The penalty fell as `m` grew, because one
//! decode then served `m` dots. The slab is gone, and with it every per-worker
//! `Vec`: a row group allocates nothing, and the only memory traffic left is
//! the activation reads the dot needs anyway.
//!
//! The dot stays a WHOLE tile wide. Narrowing it to the group instead cost
//! 2.70 instructions per multiply-accumulate at `m = 256` against the slab
//! shape's 1.41, and 5.25 against 1.46 for group-16 `Q6S16D_T64`: every call
//! pays one accumulator init and one horizontal reduction, so a half or
//! quarter width multiplies that fixed cost by the same factor. Decode is per
//! group; accumulation is per tile.
//!
//! # What the plane layout costs
//!
//! GGUF packs a block's codes and its scale adjacent, so a GGUF kernel reads
//! one contiguous run per block. TCF packs whole planes over the whole tensor
//! (SPECIFICATION.md Section 14), so decoding a range touches up to five
//! separated read streams instead of one, indexed out of the payload by
//! `tcf-core` — the cost is the extra streams, never a copy. CONFORMANCE.md
//! Section 8.1 leaves code-plane ordering unfrozen and settles it by
//! benchmark, and this kernel is where that measurement is taken.
//!
//! # Where the vector work is
//!
//! Two places, both resolved once per row group. Reconstruction runs through
//! [`Decoder`], which applies Section 13.0's arithmetic eight elements at a
//! time, bit-identically to `tcf-core`'s scalar definition. The dot is
//! [`select_dot_f32`], accumulating in vector lanes.
//!
//! The dot is the one place this kernel does NOT reproduce the reference bit
//! for bit: lane accumulation reorders a float sum, and FMA rounds a
//! multiply-add once instead of twice. Both are summation-order effects on an
//! already-approximate reduction, so the gate below compares within tolerance,
//! as it did before the vector path existed. The order is otherwise the slab
//! shape's exactly — one 64-element dot per tile per row. The reconstruction
//! stays exact: `(d, m)` arrives resolved from `tcf-core`, and no scale is
//! folded into an activation or hoisted out of a dot.

use rayon::prelude::*;
use tcf_core::{GroupCodes, TcfError, for_each_group};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;
use crate::quant::cpu::kernels::simd::tcf_decode::Decoder;

use super::super::simd::dot_f32::{DotF32, select_dot_f32};

/// Elements one reconstructed tile can hold, and the whole working set below
/// the output. v1 fixes the execution tile at 64 elements (SPECIFICATION.md
/// Section 12.1), and a group never exceeds the tile it sits in, so this
/// bounds both. A wider tile is refused, never truncated.
const TILE_SCRATCH: usize = 64;

/// The rejection every guard inside the group walk raises. [`for_each_group`]'s
/// closure returns [`TcfError`], so a guard there carries no boostr message —
/// the walk is named once, where its result is mapped.
const GEOMETRY: TcfError = TcfError::InvalidQuantGeometry;

/// Execution tiles the range-at-a-time decode shape used to take at once.
/// The kernel below no longer chunks: it asks for one row group's whole tile
/// range in a single [`for_each_group`] call, whose first tile `rows_per_group`
/// already lands on a super-block. Nothing outside this module's tests reads
/// the constant now, and being public it is kept rather than removed here.
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
    if per_block == 0 {
        return Err(Error::QuantError {
            reason: format!("{}: super-block width is zero", encoding.name()),
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
    // One CPU feature probe for the whole matmul, not one per tile.
    let dot = select_dot_f32();
    staging
        .par_chunks_mut(slab_stride)
        .enumerate()
        .try_for_each(|(group, slab)| {
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
                dot,
                slab,
            )
        })?;

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
/// dense buffer and the caller scatters it into the strided output once. Every
/// write here is local, and no shared mutable state is needed. `slab` arrives
/// zeroed and is accumulated into; `dot` is the caller's already-selected f32
/// dot, so no worker probes CPU features per group.
///
/// # Decode per group, accumulate per tile
///
/// A group is narrower than a tile — 32 elements, or 16 under `Q6S16D_T64` —
/// so each is decoded into its own slot of the tile scratch, and the row loop
/// runs only once the slots sum to the tile width. The dot therefore keeps the
/// 64-element width the tile-at-a-time shape gave it, and the accumulation
/// order is that shape's exactly.
///
/// # How a tile finds its place
///
/// [`for_each_group`] yields the range's groups in ascending order and never
/// splits one across a tile, and `first_tile` is the first tile of row
/// `row_start`. Every weight row is a whole number of tiles — `k` is a multiple
/// of the tile width, checked by the caller — so `elements`, advanced one whole
/// tile at a time, is an element offset from that row's start. Its quotient by
/// `k` is the slab row and its remainder the column, the `weight_row`/`column`
/// pair a tile index gave before. A row boundary needs no case of its own: a
/// row's last tile ends on a multiple of `k`, so the next quotient steps by one
/// and the remainder restarts at zero.
fn row_group(
    act: &[f32],
    payload: &[u8],
    at: RowGroup,
    encoding: TcfEncoding,
    dot: DotF32,
    slab: &mut [f32],
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

    // One feature probe per row group, not per group: a row group resolves
    // thousands of groups, and a probe is an atomic load no branch predictor
    // removes.
    let decoder = Decoder::detect();
    // The whole working set below the output, on the stack. A tile is read by
    // the dots below and then overwritten, so it never leaves the core.
    let mut scratch = [0.0f32; TILE_SCRATCH];
    // Elements of the tile in progress already reconstructed. Zero means no
    // tile is open, and reaching `at.tile` closes one.
    let mut filled = 0usize;
    // Which tile they belong to, read only while a tile is open.
    let mut current = 0u32;
    // Elements consumed by CLOSED tiles, from row `row_start`'s first.
    let mut elements = 0usize;

    for_each_group(
        payload,
        at.total_tiles,
        u64::try_from(first_tile).map_err(|_| overflow(encoding))?,
        u64::try_from(group_tiles).map_err(|_| overflow(encoding))?,
        layout,
        |tile_index, codes, values| {
            let width = codes.len();
            let end = filled.checked_add(width).ok_or(GEOMETRY)?;
            // A group that would overrun the tile it sits in is refused, not
            // reconstructed in part.
            if width == 0 || end > at.tile {
                return Err(GEOMETRY);
            }
            // The boundary, from `tcf-core`'s own tile index rather than an
            // assumed groups-per-tile. `filled == 0` opens a tile; a later
            // group naming a different one means the open tile's groups never
            // summed to the width, so its scratch is short and is refused
            // rather than dotted.
            if filled == 0 {
                current = tile_index;
            } else if tile_index != current {
                return Err(GEOMETRY);
            }
            let slots = scratch.get_mut(filled..end).ok_or(GEOMETRY)?;
            // `values` arrives resolved: Section 13.3's and Section 13.4's
            // two-level products ran inside `tcf-core`. What is left is
            // Section 13.0's `d * f32(q)` and Section 13.0.1's `d * f32(u) + m`,
            // through the decoder the whole-tensor path uses. No scale is
            // folded into an activation, none hoisted out of the dot below.
            match codes {
                GroupCodes::Signed(q) => decoder.signed(q, values.scale, slots),
                GroupCodes::Unsigned(u) => {
                    let min = values.min.ok_or(GEOMETRY)?;
                    decoder.unsigned(u, values.scale, min, slots);
                }
            }
            filled = end;
            if filled < at.tile {
                return Ok(());
            }

            let weights = scratch.get(..at.tile).ok_or(GEOMETRY)?;
            let start = elements;
            elements = start.checked_add(at.tile).ok_or(GEOMETRY)?;
            let slab_column = start / at.k;
            let column = start % at.k;
            // A tile that would run off the end of its own row means the
            // geometry and `k` have drifted apart.
            if column.checked_add(at.tile).is_none_or(|stop| stop > at.k) {
                return Err(GEOMETRY);
            }
            for row in 0..at.m {
                let from = row
                    .checked_mul(at.k)
                    .and_then(|base| base.checked_add(column))
                    .ok_or(GEOMETRY)?;
                let to = from.checked_add(at.tile).ok_or(GEOMETRY)?;
                let activations = act.get(from..to).ok_or(GEOMETRY)?;
                let slot = slab.get_mut(row * rows + slab_column).ok_or(GEOMETRY)?;
                *slot += dot(activations, weights);
            }
            filled = 0;
            Ok(())
        },
    )
    .map_err(|e| tcf_error(&format!("{} fused matmul", encoding.name()), e))?;

    // The range's last tile was left open, so its groups never summed to the
    // tile width. Its contribution is missing from the slab, and a partial
    // scratch must never be dotted.
    if filled != 0 {
        return Err(short(encoding));
    }
    Ok(())
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
