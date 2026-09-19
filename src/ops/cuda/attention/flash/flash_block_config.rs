//! Flash Attention v2 block (tile) configuration tables and selection.
//!
//! The shared-memory helpers these tables are measured against live in
//! `flash_smem.rs`.

use crate::error::{Error, Result};

use super::flash_smem::{compute_bwd_smem, compute_smem, device_max_smem};

/// Block config of the unsuffixed `flash_attention_bwd_{head_dim}_{dtype}` kernels.
/// Sync with `flash_v2_bwd.cu` is enforced by `tests/flash_bwd_block_config_sync.rs`.
fn bwd_block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((128, 128)),
        64 => Some((128, 128)),
        96 => Some((64, 128)),
        128 => Some((128, 64)),
        192 => Some((64, 64)),
        256 => Some((64, 64)),
        _ => None,
    }
}

/// Block config of the `flash_attention_bwd_{head_dim}_sm_{dtype}` kernels.
/// Sync with `flash_v2_bwd.cu` is enforced by `tests/flash_bwd_block_config_sync.rs`.
///
/// Sized so the F32 backward fits in 64KB, the smallest opt-in limit on GPUs that
/// support this code path; F16/BF16 need half that and FP8 a quarter.
fn bwd_block_config_small(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((64, 64)),
        64 => Some((64, 64)),
        96 => Some((32, 32)),
        128 => Some((32, 32)),
        192 => Some((16, 16)),
        256 => Some((16, 16)),
        _ => None,
    }
}

/// Test-only accessor for [`bwd_block_config_large`], so
/// `tests/flash_bwd_block_config_sync.rs` can assert this table matches the
/// `FLASH_BWD_ENTRY` instantiations in `flash_v2_bwd.cu` without launching a
/// kernel or requiring a GPU.
#[doc(hidden)]
pub fn bwd_block_config_large_for_test(head_dim: usize) -> Option<(usize, usize)> {
    bwd_block_config_large(head_dim)
}

/// Test-only accessor for [`bwd_block_config_small`] — see
/// [`bwd_block_config_large_for_test`].
#[doc(hidden)]
pub fn bwd_block_config_small_for_test(head_dim: usize) -> Option<(usize, usize)> {
    bwd_block_config_small(head_dim)
}

/// Pick the backward block config that fits this device's opt-in shared memory.
/// Returns `(block_m, block_n, use_sm_kernel)`.
///
/// The forward `block_config` is NOT usable here: it sizes for the forward layout
/// `(BLOCK_M + 2*BLOCK_N) * (head_dim + 1)`, which is smaller than the backward's
/// `2 * (BLOCK_M + BLOCK_N) * head_dim` at every head_dim this kernel supports.
pub(super) fn bwd_block_config(head_dim: usize, elem_bytes: usize) -> Result<(usize, usize, bool)> {
    let max_smem = device_max_smem();

    if let Some((bm, bn)) = bwd_block_config_large(head_dim)
        && compute_bwd_smem(bm, bn, head_dim, elem_bytes) <= max_smem
    {
        return Ok((bm, bn, false));
    }
    if let Some((bm, bn)) = bwd_block_config_small(head_dim)
        && compute_bwd_smem(bm, bn, head_dim, elem_bytes) <= max_smem
    {
        return Ok((bm, bn, true));
    }

    let reason = match bwd_block_config_small(head_dim) {
        Some((bm, bn)) => format!(
            "flash attention backward for head_dim={} needs {} bytes of shared memory \
             (smallest block config BLOCK_M={}, BLOCK_N={}, {}-byte elements) but this GPU \
             allows at most {} bytes per block",
            head_dim,
            compute_bwd_smem(bm, bn, head_dim, elem_bytes),
            bm,
            bn,
            elem_bytes,
            max_smem
        ),
        None => format!(
            "unsupported head_dim={} for flash attention backward. Supported: 32, 64, 96, 128, 192, 256",
            head_dim
        ),
    };
    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason,
    })
}

/// Launch geometry of a register-tiled forward kernel (`flash_v2.cu`,
/// `mqa_gqa.cu`): one entry of the kernel's instantiation table, plus which
/// of its two symbols it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::ops::cuda::attention) struct RegisterTile {
    /// Query rows one block owns: `WARPS * R * 32 / G` in the kernel.
    pub rows: usize,
    /// `block_dim.x`: `WARPS * 32`.
    pub threads: usize,
    /// Keys staged per shared-memory tile.
    pub block_n: usize,
    /// Selects the `_sm` symbol (two warps per block) over the four-warp one.
    pub small: bool,
}

/// Warps per block of the unsuffixed and `_sm` symbols of every
/// register-tiled forward kernel.
const REGISTER_TILE_WARPS_LARGE: usize = 4;
const REGISTER_TILE_WARPS_SMALL: usize = 2;

/// Pick between the four-warp and two-warp symbols by device fill.
///
/// Both symbols run the same kernel; they differ only in warps per block, so
/// in rows per block. The kernel keeps Q and O in registers and stages only
/// K/V, so shared memory never decides this. Two things trade: the four-warp
/// tile amortizes each staged K/V tile over twice the rows, while the
/// two-warp tile doubles the block count, which balances the last wave of
/// the grid (causal blocks differ in work) and fills an underfilled device.
/// The grid has `batch_heads * seq_len_q.div_ceil(rows)` blocks; below
/// `small_max_blocks_per_unit` four-warp blocks per compute unit the two-warp
/// tile is used. Each kernel family measures its own threshold.
///
/// `lanes` is the kernel's `G` at this head_dim and `rows_per_group` its `R`.
#[allow(clippy::too_many_arguments)]
pub(in crate::ops::cuda::attention) fn pick_register_tile(
    lanes: usize,
    rows_per_group: usize,
    block_n: usize,
    seq_len_q: usize,
    batch_heads: usize,
    compute_units: usize,
    small_max_blocks_per_unit: usize,
) -> RegisterTile {
    let tile = |warps: usize, small: bool| RegisterTile {
        rows: warps * (32 / lanes) * rows_per_group,
        threads: warps * 32,
        block_n,
        small,
    };
    let large = tile(REGISTER_TILE_WARPS_LARGE, false);
    let large_blocks = batch_heads * seq_len_q.div_ceil(large.rows);
    if large_blocks < compute_units * small_max_blocks_per_unit {
        return tile(REGISTER_TILE_WARPS_SMALL, true);
    }
    large
}

/// Dynamic shared memory of a register-tiled forward kernel: K and V tiles
/// staged as f32, whatever the tensor dtype.
pub(in crate::ops::cuda::attention) fn register_tile_smem_bytes(
    tile: RegisterTile,
    head_dim: usize,
) -> usize {
    2 * tile.block_n * head_dim * 4
}

/// `(G, R)` of the `flash_v2.cu` forward at each head_dim: lanes per query
/// row and rows per lane group. Must stay in sync with the `FLASH_FWD_DTYPE`
/// table in `flash_v2.cu`.
fn flash_fwd_group(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 | 64 => Some((4, 4)),
        96 | 128 => Some((8, 4)),
        192 | 256 => Some((8, 2)),
        _ => None,
    }
}

/// Keys per staged K/V tile (`BLOCK_N`) in every `flash_v2.cu` forward
/// instantiation; mirrors `FLASH_FWD_BLOCK_N` there.
const FLASH_FWD_BLOCK_N: usize = 16;

/// Four-warp blocks per compute unit below which the F32 `flash_v2.cu`
/// forward uses the two-warp tile.
///
/// `examples/cuda_short_query_profile.rs --flash` measured both tiles at
/// head_dim 96 and 256, causal and not, MHA and GQA, S from 64 to 2048.
/// At F32 the two-warp tile wins or ties below this fill and loses above it,
/// the same crossover [`super::super::mqa_gqa::block_config`] measured for
/// the MQA/GQA kernel. At F16 and BF16 the two-warp tile never wins: half
/// the threads stage the same K/V tile, and that staging is a larger share
/// of the half-precision kernel's time, so those dtypes always take the
/// four-warp tile ([`FLASH_FWD_HALF_SMALL_TILE_MAX_BLOCKS_PER_UNIT`]).
const FLASH_FWD_F32_SMALL_TILE_MAX_BLOCKS_PER_UNIT: usize = 8;
const FLASH_FWD_HALF_SMALL_TILE_MAX_BLOCKS_PER_UNIT: usize = 0;

/// Pick the `flash_v2.cu` forward tile for this shape. `elem_bytes` is the
/// tensor dtype size; see the two thresholds above for why it matters.
pub(super) fn flash_fwd_tile(
    head_dim: usize,
    elem_bytes: usize,
    seq_len_q: usize,
    batch_heads: usize,
    compute_units: usize,
) -> Result<RegisterTile> {
    let Some((lanes, rows_per_group)) = flash_fwd_group(head_dim) else {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for flash attention forward. Supported: 32, 64, 96, 128, 192, 256",
                head_dim
            ),
        });
    };
    let small_max_blocks_per_unit = if elem_bytes == 4 {
        FLASH_FWD_F32_SMALL_TILE_MAX_BLOCKS_PER_UNIT
    } else {
        FLASH_FWD_HALF_SMALL_TILE_MAX_BLOCKS_PER_UNIT
    };
    Ok(pick_register_tile(
        lanes,
        rows_per_group,
        FLASH_FWD_BLOCK_N,
        seq_len_q,
        batch_heads,
        compute_units,
        small_max_blocks_per_unit,
    ))
}

/// Test-only accessor for [`flash_fwd_tile`]: `(rows, threads, small)`, so
/// `tests/flash_v2_fwd_parity_cuda.rs` can check which symbol its
/// shapes select without launching a kernel.
#[doc(hidden)]
pub fn flash_fwd_tile_for_test(
    head_dim: usize,
    elem_bytes: usize,
    seq_len_q: usize,
    batch_heads: usize,
    compute_units: usize,
) -> Option<(usize, usize, bool)> {
    flash_fwd_tile(head_dim, elem_bytes, seq_len_q, batch_heads, compute_units)
        .ok()
        .map(|t| (t.rows, t.threads, t.small))
}

/// Standard (large) block config of the ONE-THREAD-PER-ROW forward kernels
/// that still stage Q/K/V in the tensor dtype: `flash_v2_fp8.cu`. The F32/F16/
/// BF16 forward in `flash_v2.cu` is register-tiled and picks its launch
/// geometry with [`flash_fwd_tile`] instead. [`block_config_small`] is the
/// fallback when the device's shared memory does not fit this.
fn block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((128, 128)),
        64 => Some((128, 128)),
        96 => Some((64, 128)),
        128 => Some((128, 64)),
        192 => Some((64, 64)),
        256 => Some((64, 64)),
        _ => None,
    }
}

/// Small-memory block config — works on GPUs with <=100KB shared memory.
/// These have corresponding `_sm` kernel variants in flash_v2.cu.
fn block_config_small(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        96 => Some((32, 32)),
        128 => Some((64, 32)),
        192 => Some((32, 16)),
        256 => Some((16, 16)),
        _ => None,
    }
}

/// Block config of the one-thread-per-row forward kernels for a head
/// dimension, accounting for device shared memory limits and the query
/// tile's row count. Returns (block_m, block_n, use_sm_kernel).
///
/// Consumed by `validate_qkv` for every forward and by the FP8 forward
/// launcher (`flash_v2_fp8.cu`, which instantiates only the large config).
/// The F32/F16/BF16 forward ignores the result and calls [`flash_fwd_tile`].
///
/// Two independent gates, in order:
///
/// 1. Shared-memory CAPABILITY (hard): the large config is only a candidate when it
///    fits `device_max_smem()`. If the large config does not fit, the small one is
///    tried, and if neither fits this returns an error.
/// 2. A `seq_len_q` PERFORMANCE rule (soft): the grid launches
///    `seq_len_q.div_ceil(block_m)` row tiles per (batch, head), and the kernel does a
///    full `BLOCK_M`-row tile of work regardless of how many rows are real. When the
///    large config fits but `seq_len_q` is small, most of its `BLOCK_M` rows go to
///    waste; the small config wastes fewer. The boundary is
///    `seq_len_q <= small_block_m`. This step only ever downgrades large -> small,
///    only when a small config exists for this head_dim and it also fits, and never
///    overrides the capability gate in step 1.
pub(super) fn block_config(
    head_dim: usize,
    elem_bytes: usize,
    seq_len_q: usize,
) -> Result<(usize, usize, bool)> {
    // Try large config first
    if let Some((bm, bn)) = block_config_large(head_dim) {
        let smem = compute_smem(bm, bn, head_dim, elem_bytes);
        if smem <= device_max_smem() {
            // Large config fits. Check whether the seq_len_q heuristic above prefers
            // the small config instead, purely to cut wasted masked rows.
            if let Some((small_bm, small_bn)) = block_config_small(head_dim)
                && seq_len_q <= small_bm
                && compute_smem(small_bm, small_bn, head_dim, elem_bytes) <= device_max_smem()
            {
                return Ok((small_bm, small_bn, true));
            }
            return Ok((bm, bn, false));
        }
    }

    // Fall back to small-memory config
    if let Some((bm, bn)) = block_config_small(head_dim) {
        let smem = compute_smem(bm, bn, head_dim, elem_bytes);
        if smem <= device_max_smem() {
            return Ok((bm, bn, true));
        }
    }

    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason: format!(
            "unsupported head_dim={} for this GPU (max shared memory: {}KB). Supported: 32, 64, 96, 128, 192, 256",
            head_dim,
            device_max_smem() / 1024
        ),
    })
}

#[cfg(test)]
mod flash_fwd_tile_tests {
    use super::*;

    #[test]
    fn f32_underfilled_grid_takes_the_two_warp_tile() {
        // 16 heads, 24 rows at head_dim 96: 16 four-warp blocks over 28 units.
        let tile = flash_fwd_tile(96, 4, 24, 16, 28).unwrap();
        assert!(tile.small);
        assert_eq!(tile.threads, 64);
        assert_eq!(tile.rows, 32);
    }

    #[test]
    fn f32_filled_grid_keeps_the_four_warp_tile() {
        // 8 heads, 2048 rows at head_dim 256: 512 four-warp blocks over 28 units.
        let tile = flash_fwd_tile(256, 4, 2048, 8, 28).unwrap();
        assert!(!tile.small);
        assert_eq!(tile.threads, 128);
        assert_eq!(tile.rows, 32);
    }

    #[test]
    fn half_precision_never_takes_the_two_warp_tile() {
        // Same underfilled grid as the F32 case above, at a 2-byte dtype.
        let tile = flash_fwd_tile(96, 2, 24, 16, 28).unwrap();
        assert!(!tile.small);
        assert_eq!(tile.rows, 64);
    }

    #[test]
    fn rows_follow_the_group_table() {
        // (G, R) = (4, 4): 8 groups of 4 rows per warp.
        assert_eq!(flash_fwd_tile(32, 4, 4096, 64, 1).unwrap().rows, 128);
        assert_eq!(flash_fwd_tile(64, 4, 4096, 64, 1).unwrap().rows, 128);
        // (8, 4): 4 groups of 4 rows per warp.
        assert_eq!(flash_fwd_tile(96, 4, 4096, 64, 1).unwrap().rows, 64);
        assert_eq!(flash_fwd_tile(128, 4, 4096, 64, 1).unwrap().rows, 64);
        // (8, 2): 4 groups of 2 rows per warp.
        assert_eq!(flash_fwd_tile(192, 4, 4096, 64, 1).unwrap().rows, 32);
        assert_eq!(flash_fwd_tile(256, 4, 4096, 64, 1).unwrap().rows, 32);
    }

    #[test]
    fn smem_is_two_f32_tiles() {
        let tile = flash_fwd_tile(256, 4, 4096, 64, 1).unwrap();
        assert_eq!(register_tile_smem_bytes(tile, 256), 2 * 16 * 256 * 4);
    }

    #[test]
    fn unsupported_head_dim_errors() {
        assert!(flash_fwd_tile(80, 4, 4096, 1, 1).is_err());
    }
}

#[cfg(test)]
mod block_config_tests {
    use super::*;
    use numr::runtime::cuda::is_cuda_available;

    // elem_bytes=4 (F32) for all cases below; head_dim=96 has a small config
    // ((32, 32)) and head_dim=64 does not, per block_config_small. Each test
    // that exercises a real smem fit checks the fit at runtime and skips
    // rather than assume a specific device's opt-in shared-memory limit.

    /// Same gate the CUDA integration tests use: the `cuda` feature can be on
    /// while no device is present, and the suite must skip, not fail.
    fn require_cuda() -> bool {
        if !is_cuda_available() {
            eprintln!("CUDA feature enabled but runtime unavailable, skipping");
            return false;
        }
        true
    }

    #[test]
    fn short_seq_len_q_prefers_small_config_when_both_fit() {
        if !require_cuda() {
            return;
        }
        let (large_bm, large_bn) = block_config_large(96).expect("head_dim 96 has a large config");
        let (small_bm, small_bn) = block_config_small(96).expect("head_dim 96 has a small config");
        let max_smem = device_max_smem();
        if compute_smem(large_bm, large_bn, 96, 4) > max_smem
            || compute_smem(small_bm, small_bn, 96, 4) > max_smem
        {
            eprintln!("device shared memory too small for this precondition, skipping");
            return;
        }
        let (block_m, _block_n, use_sm_kernel) = block_config(96, 4, 2).unwrap();
        assert!(use_sm_kernel);
        assert_eq!(block_m, small_bm);
    }

    #[test]
    fn long_seq_len_q_keeps_large_config() {
        if !require_cuda() {
            return;
        }
        let (large_bm, large_bn) = block_config_large(96).expect("head_dim 96 has a large config");
        if compute_smem(large_bm, large_bn, 96, 4) > device_max_smem() {
            eprintln!("device shared memory too small for this precondition, skipping");
            return;
        }
        let (block_m, _block_n, use_sm_kernel) = block_config(96, 4, 4096).unwrap();
        assert!(!use_sm_kernel);
        assert_eq!(block_m, large_bm);
    }

    #[test]
    fn head_dim_without_small_config_keeps_large_at_short_seq_len_q() {
        if !require_cuda() {
            return;
        }
        // head_dim=64 has no entry in block_config_small, so the seq_len_q
        // heuristic has nothing to downgrade to and must not error out.
        assert!(block_config_small(64).is_none());
        let (large_bm, large_bn) = block_config_large(64).expect("head_dim 64 has a large config");
        if compute_smem(large_bm, large_bn, 64, 4) > device_max_smem() {
            eprintln!("device shared memory too small for this precondition, skipping");
            return;
        }
        let (block_m, _block_n, use_sm_kernel) = block_config(64, 4, 2).unwrap();
        assert!(!use_sm_kernel);
        assert_eq!(block_m, large_bm);
    }

    #[test]
    fn smem_forcing_to_small_config_is_unchanged_by_seq_len_q() {
        if !require_cuda() {
            return;
        }
        // head_dim=256 large needs ~193KB, over the 164KB opt-in tier and under
        // the 227KB one, so it is refused on most devices — meaning this
        // precondition, not the seq_len_q heuristic, is what is under test:
        // capability forcing must win regardless of seq_len_q.
        let (large_bm, large_bn) =
            block_config_large(256).expect("head_dim 256 has a large config");
        let (small_bm, small_bn) =
            block_config_small(256).expect("head_dim 256 has a small config");
        let max_smem = device_max_smem();
        if compute_smem(large_bm, large_bn, 256, 4) <= max_smem {
            eprintln!("device shared memory fits the large config here, skipping");
            return;
        }
        if compute_smem(small_bm, small_bn, 256, 4) > max_smem {
            eprintln!("device shared memory too small for the small config too, skipping");
            return;
        }
        // A long seq_len_q would normally keep the large config, but it does
        // not fit here, so the small config is forced either way.
        let (block_m, _block_n, use_sm_kernel) = block_config(256, 4, 4096).unwrap();
        assert!(use_sm_kernel);
        assert_eq!(block_m, small_bm);
    }

    #[test]
    fn unsupported_head_dim_still_errors() {
        // No config exists for head_dim=999 at any seq_len_q, so this must
        // error regardless of the device's shared-memory limit.
        assert!(block_config(999, 4, 4096).is_err());
    }
}
