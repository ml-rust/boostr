//! Split count for the decode attention grid.
//!
//! The whole-sequence decode grid is one block per `(batch, head)` pair and does
//! not depend on `seq_len_k`, so a small batch runs the same handful of blocks
//! whether the KV cache holds hundreds or tens of thousands of positions. At
//! batch 1 that grid is the head count alone, which leaves most of a device
//! idle at exactly the point where a decode step reads the most memory.
//!
//! Cutting the KV sequence into `splits` slices widens the grid by that factor.
//! Each slice keeps its own `(m, l)` softmax statistics, so a combine pass
//! merges them exactly; this module owns only the choice of `splits`.

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

/// Fewest KV positions a slice may own.
///
/// Below this the per-slice prologue and the combine pass cost more than the
/// widened grid returns, and the partial buffers grow for nothing.
///
/// Measured: this bound, not the device-fill target, is what selects the split
/// count at short context, and a short cache was leaving the device at well
/// under one wave. A slice still gives every warp of its block several
/// positions to walk.
const DECODE_MIN_CHUNK: usize = 32;

/// Upper bound on the split count.
///
/// The combine kernel walks the slices serially, and the partial buffers scale
/// with this, so the widening stops once the device is comfortably full.
const DECODE_MAX_SPLITS: usize = 32;

/// Number of KV slices to cut the sequence into, or `1` to keep the
/// whole-sequence launch.
///
/// `base_blocks` is the un-split grid width (`batch * num_heads`), `kv_len`
/// the number of KV positions the kernel will actually read, and `head_dim`
/// the block width in threads.
#[inline]
pub(super) fn decode_split_count(
    device_index: usize,
    base_blocks: usize,
    kv_len: usize,
    head_dim: usize,
) -> usize {
    // CudaDevice::new is a zero-cost index wrapper; profile() reads the cached
    // profile, so this is an atomic load rather than a driver query.
    let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
    decode_split_for_units(compute_units, base_blocks, kv_len, head_dim)
}

/// The split rule itself, separated from the device query so it is testable
/// without a device.
#[inline]
fn decode_split_for_units(
    compute_units: usize,
    base_blocks: usize,
    kv_len: usize,
    head_dim: usize,
) -> usize {
    let warps_per_block = (head_dim / DECODE_LANES).max(1);
    let blocks_per_unit = (DECODE_WARPS_PER_UNIT / warps_per_block).max(DECODE_BLOCKS_PER_UNIT);
    // An unknown profile reports zero compute units. The target is then zero,
    // no grid underfills it, and every shape keeps the whole-sequence launch.
    let target_blocks = compute_units.saturating_mul(blocks_per_unit);
    if base_blocks == 0 || base_blocks >= target_blocks {
        return 1;
    }

    let max_splits = (kv_len / DECODE_MIN_CHUNK).min(DECODE_MAX_SPLITS);
    if max_splits < 2 {
        return 1;
    }

    target_blocks.div_ceil(base_blocks).clamp(1, max_splits)
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

    #[test]
    fn unknown_profile_never_splits() {
        assert_eq!(decode_split_for_units(0, 32, 1 << 20, 128), 1);
    }

    #[test]
    fn grid_that_already_fills_the_device_is_left_alone() {
        // 512 blocks against a target of 28 * 8 = 224.
        assert_eq!(decode_split_for_units(28, 512, 1 << 20, 128), 1);
    }

    #[test]
    fn short_sequence_is_left_alone() {
        // Fewer than two minimum chunks, so there is nothing to split.
        assert_eq!(
            decode_split_for_units(28, 32, 2 * DECODE_MIN_CHUNK - 1, 128),
            1
        );
    }

    #[test]
    fn narrow_grid_over_long_sequence_fills_the_device() {
        let splits = decode_split_for_units(28, 32, 16384, 128);
        assert!(splits > 1, "splits={splits}");
        assert!(
            32 * splits >= 28 * DECODE_BLOCKS_PER_UNIT,
            "splits={splits}"
        );
    }

    #[test]
    fn split_count_never_starves_a_slice() {
        let kv_len = 1024;
        let splits = decode_split_for_units(28, 1, kv_len, 128);
        assert!(kv_len / splits >= DECODE_MIN_CHUNK, "splits={splits}");
    }

    /// A one-warp block gets the warp target, a four-warp block the block
    /// floor, and a wider block never drops below the floor.
    #[test]
    fn narrow_head_widens_the_grid() {
        let wide = decode_split_for_units(28, 32, 1 << 20, 128);
        let narrow = decode_split_for_units(28, 32, 1 << 20, 32);
        let widest = decode_split_for_units(28, 32, 1 << 20, 256);
        assert_eq!(wide * 32, 28 * DECODE_BLOCKS_PER_UNIT);
        assert_eq!(narrow * 32, 28 * DECODE_WARPS_PER_UNIT);
        assert_eq!(widest, wide);
    }

    #[test]
    fn window_bounds_the_span() {
        assert_eq!(decode_kv_span(8192, 0), 8192);
        assert_eq!(decode_kv_span(8192, 1024), 1024);
        assert_eq!(decode_kv_span(512, 1024), 512);
    }

    #[test]
    fn split_count_stays_within_the_cap() {
        assert!(decode_split_for_units(1024, 1, 1 << 20, 32) <= DECODE_MAX_SPLITS);
    }
}
