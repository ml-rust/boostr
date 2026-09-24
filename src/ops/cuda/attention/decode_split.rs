//! Split count for the decode attention grid.
//!
//! The whole-sequence decode grid is one block per `(batch, head)` pair and does
//! not depend on `seq_len_k`, so a small batch runs the same handful of blocks
//! whether the KV cache holds hundreds or tens of thousands of positions. At
//! batch 1 that grid is the head count alone, which leaves most of a device
//! idle at exactly the point where a decode step reads the most memory.
//!
//! Cutting the KV sequence into slices widens the grid by the slice count.
//! Each slice keeps its own `(m, l)` softmax statistics, so a combine pass
//! merges them exactly, in slice order. This module owns the cut: each row
//! cuts its own span by a rule of the head count, head dimension and device,
//! never of the batch size or of the padding around it, so a row is cut the
//! same way whatever it is batched with.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::CudaDevice;

/// Fewest resident blocks per compute unit the split grid aims for.
///
/// The decode kernel uses `head_dim` threads and no shared memory beyond a few
/// floats, so occupancy is bounded by the resident block limit rather than by
/// registers. Filling several blocks per unit keeps enough loads in flight to
/// cover DRAM latency without shrinking each slice into launch overhead.
const DECODE_BLOCKS_PER_UNIT: usize = 8;

/// Resident warps per compute unit the split grid aims for at a narrow head.
///
/// A block is one warp per 32 head dimensions, so at `head_dim` 32 the block
/// floor above leaves a unit with eight warps in flight and the kernel
/// latency-bound. The target is raised to this many warps, divided by the
/// warps per block, whenever that exceeds the block floor; at `head_dim` 128
/// and above the two rules agree.
const DECODE_WARPS_PER_UNIT: usize = 32;

/// Lanes per warp, matching `DECODE_LANES` in `kernels/attention/decode_attention.cu`.
const DECODE_LANES: usize = 32;

/// Fewest KV positions a slice may own, and the quantum slice lengths are
/// rounded to.
///
/// Below this the per-slice prologue and the combine pass cost more than the
/// widened grid returns, and the partial buffers grow for nothing. A slice
/// still gives every warp of its block several positions to walk.
const DECODE_MIN_CHUNK: usize = 32;

/// Upper bound on the split count.
///
/// The combine kernel walks the slices serially, and the partial buffers scale
/// with this, so the widening stops once the device is comfortably full.
const DECODE_MAX_SPLITS: usize = 32;

/// How the decode grid cuts the KV span.
///
/// Every row cuts its own span, from its first key, into slices of the same
/// length: `ceil(span / want)` rounded up to [`DECODE_MIN_CHUNK`], where
/// `want` is [`Self::fill`] clamped to what the span allows. The kernel
/// derives that per row from `fill` and the row's padding start
/// (`decode_row_slice` in `kernels/attention/decode_attention.cu`); the host
/// applies the same rule to the longest span to size the grid. Nothing in
/// it reads the batch size, so a row's slices, and so the float sequence its
/// output is formed from, are the same whether it decodes alone or padded
/// in a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecodeSlices {
    /// Slices to launch per row; `1` keeps the whole-sequence kernel. The
    /// longest row's own count, which bounds every shorter row's.
    pub splits: usize,
    /// Slices one row wants when its span allows them: the device fill
    /// target over the head count. The kernel's argument.
    pub fill: usize,
}

/// The slices a `kv_len`-position span is cut into for a model with
/// `num_heads` query heads at `head_dim`.
#[inline]
pub(super) fn decode_slices(
    device_index: usize,
    num_heads: usize,
    kv_len: usize,
    head_dim: usize,
) -> DecodeSlices {
    // CudaDevice::new is a zero-cost index wrapper; profile() reads the cached
    // profile, so this is an atomic load rather than a driver query.
    let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
    decode_slices_for_units(compute_units, num_heads, kv_len, head_dim)
}

/// Slices a `span`-key row cuts itself into when it wants `fill`: `fill`
/// clamped to `[1, min(span / DECODE_MIN_CHUNK, DECODE_MAX_SPLITS)]`, and
/// `1` when fewer than two minimum chunks exist. The kernel's
/// `decode_row_slice` applies the same rule per row.
const fn row_splits(span: usize, fill: usize) -> usize {
    let max_splits = span / DECODE_MIN_CHUNK;
    let max_splits = if max_splits > DECODE_MAX_SPLITS {
        DECODE_MAX_SPLITS
    } else {
        max_splits
    };
    if max_splits < 2 {
        return 1;
    }
    if fill < 1 {
        1
    } else if fill > max_splits {
        max_splits
    } else {
        fill
    }
}

/// Fewest KV pages a paged slice may own: [`DECODE_MIN_CHUNK`] keys rounded
/// up to whole pages. The paged decode path cuts in page units so a slice
/// boundary lands on a block-table entry and never straddles a page, and the
/// page-unit quantum is the token-unit one divided by the page length rather
/// than a number of its own.
pub(super) const fn paged_min_chunk_blocks(block_size: usize) -> usize {
    let page = if block_size == 0 { 1 } else { block_size };
    let chunk = DECODE_MIN_CHUNK.div_ceil(page);
    if chunk == 0 { 1 } else { chunk }
}

/// Slices a row owning `blocks` KV pages cuts itself into when it wants
/// `fill`: `fill` clamped to `[1, min(blocks / chunk, DECODE_MAX_SPLITS)]` for
/// the `chunk` [`paged_min_chunk_blocks`] gives, and `1` when fewer than two
/// chunks exist. [`row_splits`] in page units. `paged_decode_row_slice` in
/// `kernels/attention/paged_decode_attention.cu` applies the same rule to each
/// row's own page count, so the host sizes the grid from the block-table
/// capacity and every shorter row's trailing slices come out empty.
pub(super) const fn paged_row_splits(blocks: usize, fill: usize, block_size: usize) -> usize {
    let chunk = paged_min_chunk_blocks(block_size);
    let max_splits = blocks / chunk;
    let max_splits = if max_splits > DECODE_MAX_SPLITS {
        DECODE_MAX_SPLITS
    } else {
        max_splits
    };
    if max_splits < 2 {
        return 1;
    }
    if fill < 1 {
        1
    } else if fill > max_splits {
        max_splits
    } else {
        fill
    }
}

/// Test-only accessor for the per-row-ends split count: composes
/// [`decode_slices`] and [`paged_row_splits`] exactly as
/// `paged_decode_attention_fwd` and `paged_decode_attention_fwd_graph` do
/// when a per-row key end is present, so
/// `tests/paged_attention_batch_invariance_cuda.rs` can assert which grid a
/// fixture exercises without reaching into either launch site. `1` keeps the
/// whole-sequence kernel, more than `1` takes the split grid.
#[doc(hidden)]
pub fn paged_decode_splits(
    device_index: usize,
    num_heads: usize,
    max_num_blocks: usize,
    block_size: usize,
    head_dim: usize,
) -> usize {
    let span = max_num_blocks * block_size;
    let fill = decode_slices(device_index, num_heads, span, head_dim).fill;
    paged_row_splits(max_num_blocks, fill, block_size)
}

/// The slice rule itself, separated from the device query so it is testable
/// without a device.
///
/// `fill` is `ceil(target / num_heads)`, the slices a single row needs to
/// put `compute_units * blocks_per_unit` blocks in flight, capped at
/// [`DECODE_MAX_SPLITS`]. The launched count is the longest row's
/// [`row_splits`]; a batch launches `batch` times as many blocks and every
/// row is cut the same way it is alone.
#[inline]
fn decode_slices_for_units(
    compute_units: usize,
    num_heads: usize,
    kv_len: usize,
    head_dim: usize,
) -> DecodeSlices {
    let warps_per_block = (head_dim / DECODE_LANES).max(1);
    let blocks_per_unit = (DECODE_WARPS_PER_UNIT / warps_per_block).max(DECODE_BLOCKS_PER_UNIT);
    // An unknown profile reports zero compute units. The target is then zero
    // and every shape keeps the whole-sequence launch.
    let target_blocks = compute_units.saturating_mul(blocks_per_unit);
    if num_heads == 0 || target_blocks == 0 {
        return DecodeSlices { splits: 1, fill: 1 };
    }
    let fill = target_blocks
        .div_ceil(num_heads)
        .clamp(1, DECODE_MAX_SPLITS);
    DecodeSlices {
        splits: row_splits(kv_len, fill),
        fill,
    }
}

/// Kernel-name dtype suffix for the decode kernels.
///
/// The decode kernels are instantiated for the three float dtypes serving uses.
/// Anything else has no decode kernel and belongs on the general path.
pub(super) fn decode_dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("fp32"),
        DType::F16 => Ok("fp16"),
        DType::BF16 => Ok("bf16"),
        other => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("decode attention supports F32/F16/BF16, got {other:?}"),
        }),
    }
}

/// Whether a decode kernel exists for `dtype`.
pub(super) fn decode_supports_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// Head dimensions the contiguous decode kernel is instantiated for, in sync
/// with the `DECODE_ATTENTION_DTYPE` list in
/// `kernels/attention/decode_attention.cu`. The same set `validate_qkv`
/// admits, so every `seq_len_q == 1` forward in a decode dtype has a kernel.
pub(super) const DECODE_HEAD_DIMS: [usize; 6] = [32, 64, 96, 128, 192, 256];

/// Whether the contiguous decode kernel is instantiated for `head_dim`.
pub(super) fn decode_supports_head_dim(head_dim: usize) -> bool {
    DECODE_HEAD_DIMS.contains(&head_dim)
}

/// KV positions a decode step reads: the whole sequence, or the window suffix
/// when one is set. This is what the split count is sized against.
pub(super) fn decode_kv_span(seq_len_k: usize, window_size: usize) -> usize {
    if window_size > 0 {
        seq_len_k.min(window_size)
    } else {
        seq_len_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SM count the cases below are written against.
    const CU: usize = 28;

    #[test]
    fn unknown_profile_never_splits() {
        assert_eq!(decode_slices_for_units(0, 32, 1 << 20, 128).splits, 1);
        assert_eq!(decode_slices_for_units(CU, 0, 1 << 20, 128).splits, 1);
    }

    #[test]
    fn short_sequence_is_left_alone() {
        // Fewer than two minimum chunks, so there is nothing to split.
        assert_eq!(
            decode_slices_for_units(CU, 1, 2 * DECODE_MIN_CHUNK - 1, 128).splits,
            1
        );
        assert_eq!(
            decode_slices_for_units(CU, 16, 2 * DECODE_MIN_CHUNK, 128).splits,
            2
        );
    }

    #[test]
    fn a_single_row_fills_the_device() {
        // Fixture `CU` (28) * 8 = 224 target blocks over 16 heads: 14
        // slices, at any span long enough to hold them.
        let s = decode_slices_for_units(CU, 16, 4096, 128);
        assert_eq!(
            s,
            DecodeSlices {
                splits: 14,
                fill: 14
            }
        );
        assert_eq!(decode_slices_for_units(CU, 16, 700, 128).splits, 14);
        // A span of 10 minimum chunks holds 10.
        assert_eq!(decode_slices_for_units(CU, 16, 320, 128).splits, 10);
    }

    #[test]
    fn the_rule_takes_the_head_count_not_the_block_count() {
        // The batch never enters: this is a function of (heads, span) alone,
        // and the kernel applies `row_splits` to each row's own span.
        for &kv in &[31usize, 700, 4096, 8192, 1 << 16] {
            let s = decode_slices_for_units(CU, 16, kv, 128);
            assert_eq!(s.splits, row_splits(kv, s.fill));
            assert!(s.splits <= DECODE_MAX_SPLITS, "kv={kv} splits={}", s.splits);
        }
    }

    /// A one-warp block gets the warp target, a four-warp block the block
    /// floor, and a wider block never drops below the floor.
    #[test]
    fn narrow_head_widens_the_grid() {
        let wide = decode_slices_for_units(CU, 32, 1 << 20, 128);
        let narrow = decode_slices_for_units(CU, 32, 1 << 20, 32);
        let widest = decode_slices_for_units(CU, 32, 1 << 20, 256);
        assert_eq!(wide.splits * 32, CU * DECODE_BLOCKS_PER_UNIT);
        assert_eq!(narrow.splits * 32, CU * DECODE_WARPS_PER_UNIT);
        assert_eq!(widest, wide);
    }

    #[test]
    fn window_bounds_the_span() {
        assert_eq!(decode_kv_span(8192, 0), 8192);
        assert_eq!(decode_kv_span(8192, 1024), 1024);
        assert_eq!(decode_kv_span(512, 1024), 512);
    }

    /// The page-unit quantum is the token-unit one over the page length, and
    /// never drops below a whole page.
    #[test]
    fn paged_chunk_is_the_token_chunk_in_pages() {
        assert_eq!(paged_min_chunk_blocks(1), DECODE_MIN_CHUNK);
        assert_eq!(paged_min_chunk_blocks(16), 2);
        assert_eq!(paged_min_chunk_blocks(DECODE_MIN_CHUNK), 1);
        assert_eq!(paged_min_chunk_blocks(256), 1);
        assert_eq!(paged_min_chunk_blocks(0), DECODE_MIN_CHUNK);
    }

    #[test]
    fn paged_row_splits_clamps_to_the_page_count() {
        // Two chunks of two pages each: the row allows two slices at most.
        assert_eq!(paged_row_splits(4, 32, 16), 2);
        // One chunk only, so nothing to split.
        assert_eq!(paged_row_splits(3, 32, 16), 1);
        assert_eq!(paged_row_splits(0, 32, 16), 1);
        // Fill below the cap passes through.
        assert_eq!(paged_row_splits(64, 5, 16), 5);
        // A page as long as the token chunk gives one page per chunk.
        assert_eq!(paged_row_splits(7, 32, 32), 7);
        assert!(paged_row_splits(1 << 20, 1 << 20, 16) <= DECODE_MAX_SPLITS);
    }

    /// A shorter row never asks for more slices than a longer one, so a grid
    /// sized from the block-table capacity covers every row in the batch.
    #[test]
    fn paged_row_splits_grows_with_the_page_count() {
        let fill = 14;
        let mut previous = 0;
        for blocks in 0..256usize {
            let splits = paged_row_splits(blocks, fill, 16);
            assert!(
                splits >= previous,
                "blocks={blocks} splits={splits} dropped below {previous}"
            );
            previous = splits;
        }
    }

    #[test]
    fn split_count_stays_within_the_cap() {
        assert!(decode_slices_for_units(1024, 1, 1 << 20, 32).splits <= DECODE_MAX_SPLITS);
        assert!(decode_slices_for_units(1024, 1, 1 << 20, 32).fill <= DECODE_MAX_SPLITS);
    }
}
