//! Runtime block-size selection for the MQA/GQA CUDA kernels.
//!
//! `mqa_gqa.cu` and `mqa_gqa_bwd.cu` each emit two block-size variants per
//! (head_dim, dtype): the unsuffixed symbol and a `_sm`-suffixed one.
//! The forward picks by device fill (`mqa_fwd_tile`); the backward picks by
//! the device's opt-in shared-memory limit (`mqa_bwd_block_config`), so a GPU
//! with a small limit still launches.

use crate::error::{Error, Result};

use super::super::flash::flash_block_config::{RegisterTile, pick_register_tile};
use super::super::flash::flash_utils::{compute_bwd_smem, device_max_smem};

/// Shared memory element size of the MQA/GQA backward kernels.
///
/// `mqa_gqa_bwd.cu` has a single backward impl, `mqa_gqa_bwd_impl`. It declares
/// `extern __shared__ float smem[]` and stages K/V/Q/dO as f32 there, converting
/// on load, so the requirement is independent of the tensor dtype.
pub(super) const BWD_SMEM_ELEM_BYTES: usize = 4;

/// Lanes per query row (`G`) at each head_dim. Must stay in sync with the
/// `MQA_GQA_FWD_DTYPE` table in `mqa_gqa.cu`.
fn mqa_fwd_lanes_per_row(head_dim: usize) -> Option<usize> {
    match head_dim {
        32 | 64 => Some(4),
        128 => Some(8),
        _ => None,
    }
}

/// Rows per lane group (`R`) in every forward instantiation.
const MQA_FWD_ROWS_PER_GROUP: usize = 4;
/// Keys per staged K/V tile (`BLOCK_N`) in every forward instantiation.
const MQA_FWD_BLOCK_N: usize = 16; // mirrors `MQA_GQA_FWD_BLOCK_N` in mqa_gqa.cu

/// Four-warp blocks per compute unit below which the two-warp tile is used.
/// `examples/cuda_short_query_profile.rs --prefill` measured the crossover
/// for this kernel; [`pick_register_tile`] explains what trades.
const MQA_FWD_SMALL_TILE_MAX_BLOCKS_PER_UNIT: usize = 8;

/// Pick the forward tile for this shape. The grid has
/// `batch_heads * seq_len_q.div_ceil(rows)` blocks; see
/// [`MQA_FWD_SMALL_TILE_MAX_BLOCKS_PER_UNIT`] for the rule.
pub(super) fn mqa_fwd_tile(
    head_dim: usize,
    seq_len_q: usize,
    batch_heads: usize,
    compute_units: usize,
) -> Result<RegisterTile> {
    let Some(lanes) = mqa_fwd_lanes_per_row(head_dim) else {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "MQA/GQA kernels support head_dim 32/64/128, got {}",
                head_dim
            ),
        });
    };
    Ok(pick_register_tile(
        lanes,
        MQA_FWD_ROWS_PER_GROUP,
        MQA_FWD_BLOCK_N,
        seq_len_q,
        batch_heads,
        compute_units,
        MQA_FWD_SMALL_TILE_MAX_BLOCKS_PER_UNIT,
    ))
}

/// Block config of the unsuffixed `mqa_gqa_bwd_{head_dim}_{dtype}` kernels.
/// Must stay in sync with the "Large blocks" instantiations in `mqa_gqa_bwd.cu`.
fn mqa_bwd_block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((128, 128)),
        64 => Some((128, 128)),
        128 => Some((128, 64)),
        _ => None,
    }
}

/// Block config of the `mqa_gqa_bwd_{head_dim}_{dtype}_sm` kernels.
/// Must stay in sync with the "Small blocks" instantiations in `mqa_gqa_bwd.cu`.
fn mqa_bwd_block_config_small(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((64, 64)),
        64 => Some((64, 32)),
        128 => Some((64, 32)),
        _ => None,
    }
}

/// Pick the MQA/GQA backward block config that fits this device's opt-in shared
/// memory. Returns `(block_m, block_n, use_sm_kernel)`; `use_sm_kernel` selects
/// the `_sm`-suffixed kernel symbol.
///
/// [`mqa_fwd_tile`] is NOT usable here: the forward stages only K/V and keeps
/// Q/O in registers, while the backward stages 4 tiles of f32, which can
/// exceed a small opt-in limit.
pub(super) fn mqa_bwd_block_config(head_dim: usize) -> Result<(usize, usize, bool)> {
    let max_smem = device_max_smem();

    if let Some((bm, bn)) = mqa_bwd_block_config_large(head_dim)
        && compute_bwd_smem(bm, bn, head_dim, BWD_SMEM_ELEM_BYTES) <= max_smem
    {
        return Ok((bm, bn, false));
    }
    if let Some((bm, bn)) = mqa_bwd_block_config_small(head_dim)
        && compute_bwd_smem(bm, bn, head_dim, BWD_SMEM_ELEM_BYTES) <= max_smem
    {
        return Ok((bm, bn, true));
    }

    let reason = match mqa_bwd_block_config_small(head_dim) {
        Some((bm, bn)) => format!(
            "MQA/GQA backward for head_dim={} needs {} bytes of shared memory \
             (smallest block config BLOCK_M={}, BLOCK_N={}, f32 staging) but this GPU \
             allows at most {} bytes per block",
            head_dim,
            compute_bwd_smem(bm, bn, head_dim, BWD_SMEM_ELEM_BYTES),
            bm,
            bn,
            max_smem
        ),
        None => format!(
            "MQA/GQA backward supports head_dim 32/64/128, got {}",
            head_dim
        ),
    };
    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason,
    })
}

/// Returns true if the dedicated MQA/GQA kernels are CAPABLE of this shape.
///
/// Both conditions below are capability limits — "the kernel cannot correctly
/// or completely handle this shape" — not a performance judgment call:
///
/// - `head_dim ∈ {32, 64, 128}`: the exact template set `.cu` instantiates.
///   See `mqa_fwd_lanes_per_row` in this file, which mirrors those
///   instantiations. Any other head_dim has no kernel symbol to call.
/// - `num_heads.is_multiple_of(num_kv_heads)`: the kernel maps
///   `kv_head_idx = q_head_idx / (num_heads / num_kv_heads)`. When that
///   division isn't exact, the mapping reads past the end of the KV heads.
///   Both call sites in `flash.rs` already run `flash_utils::validate_qkv`
///   first, which rejects a non-divisible pair before this gate runs — the
///   check here is a second, cheap guard for any other caller of this public
///   function.
///
/// There used to be a third condition, `num_heads / num_kv_heads >= 4`: a
/// performance guess from the original skeleton, gating out shapes the kernel
/// handles correctly, on the theory that the dedicated kernel only paid off
/// at extreme GQA ratios. Measurement (F32, causal, batch 1, head_dim
/// 32/64/128, seq 512/4096, ratios 1 through 32) found no crossover — the
/// dedicated kernel wins a small, flat margin at every ratio, MHA (ratio 1)
/// included, with no ratio dependence. boostr is a library: the automatic
/// route is a default, and the default follows the measurement, not a
/// guessed threshold — so this condition is gone. Capability, not performance
/// policy, is the only thing that should gate this function; do not re-add a
/// ratio floor without a new measurement showing an actual crossover.
///
/// Shape only, so this stays testable without a device. `mqa_gqa.cu` (forward)
/// has no native bf16 arithmetic and compiles at sm_75. `mqa_gqa_bwd.cu` has
/// real `__CUDA_ARCH__` guards around native bf16 arithmetic and needs sm_80.
/// Both call sites in `flash.rs` gate this behind `caps.bf16` so the forward
/// and backward stay on the same kernel family — never forward on the
/// dedicated kernel with backward on the general fallback. Below sm_80 both
/// fall back to the general flash kernel, which runs F32/F16/BF16 on Turing.
///
/// Gates PREFILL only: both call sites in `flash.rs` route `seq_len_q == 1` to
/// the decode path before reaching this check.
pub fn should_use_mqa_gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> bool {
    if num_kv_heads == 0 {
        return false;
    }
    num_heads.is_multiple_of(num_kv_heads) && matches!(head_dim, 32 | 64 | 128)
}

#[cfg(test)]
mod mqa_fwd_tile_tests {
    use super::*;

    #[test]
    fn underfilled_grid_takes_the_two_warp_tile() {
        // 16 heads, 100 rows: 32 four-warp blocks over 40 units.
        let tile = mqa_fwd_tile(128, 100, 16, 40).unwrap();
        assert!(tile.small);
        assert_eq!(tile.threads, 64);
        assert_eq!(tile.rows, 32);
    }

    #[test]
    fn filled_grid_keeps_the_four_warp_tile() {
        // 16 heads, 4096 rows: 1024 four-warp blocks over 40 units.
        let tile = mqa_fwd_tile(128, 4096, 16, 40).unwrap();
        assert!(!tile.small);
        assert_eq!(tile.threads, 128);
        assert_eq!(tile.rows, 64);
    }

    #[test]
    fn rows_follow_lanes_per_row() {
        // G = 4 at head_dim 64: 8 groups of 4 rows per warp.
        assert_eq!(mqa_fwd_tile(64, 4096, 64, 1).unwrap().rows, 128);
        assert_eq!(mqa_fwd_tile(32, 4096, 64, 1).unwrap().rows, 128);
        assert_eq!(mqa_fwd_tile(128, 4096, 64, 1).unwrap().rows, 64);
    }

    #[test]
    fn smem_is_two_f32_tiles() {
        use super::super::super::flash::flash_block_config::register_tile_smem_bytes;
        let tile = mqa_fwd_tile(128, 4096, 64, 1).unwrap();
        assert_eq!(register_tile_smem_bytes(tile, 128), 2 * 16 * 128 * 4);
    }

    #[test]
    fn unsupported_head_dim_errors() {
        assert!(mqa_fwd_tile(999, 4096, 1, 1).is_err());
    }
}
