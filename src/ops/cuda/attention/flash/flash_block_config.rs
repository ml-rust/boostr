//! Flash Attention v2 block (tile) configuration tables and selection.
//!
//! Split out of `flash_utils.rs`; the shared-memory helpers these tables are
//! measured against live in `flash_smem.rs`.

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

/// Standard (large) block config — used when the device's shared memory fits it.
/// [`block_config_small`] is the fallback when it does not.
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

/// Get block configuration for a head dimension, accounting for device shared memory
/// limits and the query tile's row count. Returns (block_m, block_n, use_sm_kernel).
///
/// Two independent gates, in order:
///
/// 1. Shared-memory CAPABILITY (hard): the large config is only a candidate when it
///    fits `device_max_smem()`. This part is unchanged from before `seq_len_q` was a
///    factor — if the large config does not fit, the small one is tried, and if
///    neither fits this returns an error.
/// 2. A `seq_len_q` PERFORMANCE rule (soft): the grid launches
///    `seq_len_q.div_ceil(block_m)` row tiles per (batch, head), and the kernel does a
///    full `BLOCK_M`-row tile of work regardless of how many rows are real. When the
///    large config fits but `seq_len_q` is small, most of its `BLOCK_M` rows go to
///    waste; the small config wastes fewer. The boundary used here is
///    `seq_len_q <= small_block_m`: at or below the small tile's own `BLOCK_M`, the
///    large tile can only be wasting rows the small tile would not, while the small
///    tile's extra K-loop iterations (smaller `BLOCK_N`, more iterations to cover the
///    same `seq_len_k`) are the cost being traded against. The rule itself is measured,
///    not guessed — see [`super::super::mqa_gqa::block_config::mqa_fwd_block_config`], which
///    carries the same boundary and the measurement behind it on the MQA/GQA forward
///    path — and the row-waste argument above applies here by the same reasoning. It
///    has NOT, however, been observed to change a selection on this path: at
///    head_dim=96, [`block_config_large`]'s shared-memory requirement (196KB+) is well
///    beyond any current device's opt-in limit, so head_dim=96 is already forced onto
///    the small config by gate 1 before this rule ever runs, and head_dim 32/64/128 no
///    longer reach this function at all — [`super::super::mqa_gqa::should_use_mqa_gqa`] in `mqa_gqa/block_config.rs`
///    routes them to the dedicated MQA/GQA kernels instead. So on the head_dims that
///    currently reach this path, the rule is correct and harmless but inert. This step
///    only ever downgrades large -> small, and only when a small config exists for this
///    head_dim and it also fits; it never overrides the capability gate in step 1.
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
