//! Runtime block-size selection for the MQA/GQA CUDA kernels.
//!
//! `mqa_gqa.cu` and `mqa_gqa_bwd.cu` each emit two block-size variants per
//! (head_dim, dtype): the unsuffixed large-block symbol and a `_sm`-suffixed
//! small-block symbol. The pickers here choose between them from the device's
//! opt-in shared-memory limit, so a GPU with a small limit still launches.

use crate::error::{Error, Result};

use super::super::flash_utils::{compute_bwd_smem, compute_smem, device_max_smem};

/// Shared memory element size of the MQA/GQA backward kernels.
///
/// `mqa_gqa_bwd.cu` has a single backward impl, `mqa_gqa_bwd_impl`. It declares
/// `extern __shared__ float smem[]` and stages K/V/Q/dO as f32 there, converting
/// on load, so the requirement is independent of the tensor dtype.
pub(super) const BWD_SMEM_ELEM_BYTES: usize = 4;

/// Block config of the unsuffixed `mqa_gqa_fwd_{head_dim}_{dtype}` kernels.
/// Must stay in sync with the "Large blocks" instantiations in `mqa_gqa.cu`.
fn mqa_fwd_block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((128, 128)),
        64 => Some((128, 128)),
        128 => Some((128, 64)),
        _ => None,
    }
}

/// Block config of the `mqa_gqa_fwd_{head_dim}_{dtype}_sm` kernels.
/// Must stay in sync with the "Small blocks" instantiations in `mqa_gqa.cu`.
///
/// Same block sizes the backward half uses at each head_dim, so the two read
/// alike. Sized so the f32 forward fits 99KB: 25344 / 33280 / 66048 bytes for
/// head_dim 32 / 64 / 128.
fn mqa_fwd_block_config_small(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((64, 64)),
        64 => Some((64, 32)),
        128 => Some((64, 32)),
        _ => None,
    }
}

/// Pick the MQA/GQA forward block config that fits this device's opt-in shared
/// memory. Returns `(block_m, block_n, use_sm_kernel)`; `use_sm_kernel` selects
/// the `_sm`-suffixed kernel symbol, and `block_m` is the launcher's `block_dim.x`.
///
/// `elem_bytes` is the shared-memory ELEMENT size, not always the tensor dtype
/// size: `mqa_gqa_fwd_fp8_impl` declares `extern __shared__ float smem[]` and
/// dequantizes on load, so FP8 needs 4 bytes per element, not 1. The FP32/FP16/
/// BF16 impls stage in the tensor dtype.
///
/// Two independent gates, in order:
///
/// 1. Shared-memory CAPABILITY (hard): the large config is only a candidate when
///    it fits `device_max_smem()`. Unchanged from before `seq_len_q` was a
///    factor — if the large config does not fit, the small one is tried, and if
///    neither fits this errors.
/// 2. A `seq_len_q` PERFORMANCE rule (soft, measured): the grid launches
///    `seq_len_q.div_ceil(block_m)` row tiles per (batch, head); the kernel does
///    a full `block_m`-row tile of work regardless of how many rows are real.
///    When the large config fits but `seq_len_q` is small, most of its rows go
///    to waste; the small config wastes fewer. Measured on this forward path
///    (head_dim 64, F32, causal, a fixed long `seq_len_k`, sweeping `seq_len_q`
///    and batch): wherever this boundary fires, the small tile wins at both
///    batch 1 and batch 8, never loses, by a wide margin at the shortest
///    queries narrowing toward the boundary. The cause is row waste, not
///    device fill — the win persists at batch 8, where the large tile's grid
///    already fills the device on its own. Widening the boundary to
///    `2 * small_block_m` was measured and rejected: it wins at batch 1 but
///    loses at batch 8, because at that width the trade stops being about row
///    waste and starts depending on device fill — batch 1 underfills the
///    device with the large tile, so more/smaller blocks help, while batch 8
///    already fills it and the small tile's extra K-loop iterations dominate.
///    So the boundary stays at `small_block_m`, which wins or ties at every
///    batch measured. A wider rule would need to consult the device, e.g.
///    comparing `batch * num_heads * seq_len_q.div_ceil(large_block_m)`
///    against `compute_units` the way `decode_split.rs` does — that is the
///    next step for the batch-1 win still on the table, not a larger constant.
///    This step only ever downgrades large -> small, only when a small config
///    exists for this head_dim and it also fits, and never overrides gate 1.
pub(super) fn mqa_fwd_block_config(
    head_dim: usize,
    elem_bytes: usize,
    seq_len_q: usize,
) -> Result<(usize, usize, bool)> {
    let max_smem = device_max_smem();

    if let Some((bm, bn)) = mqa_fwd_block_config_large(head_dim)
        && compute_smem(bm, bn, head_dim, elem_bytes) <= max_smem
    {
        if let Some((small_bm, small_bn)) = mqa_fwd_block_config_small(head_dim)
            && seq_len_q <= small_bm
            && compute_smem(small_bm, small_bn, head_dim, elem_bytes) <= max_smem
        {
            return Ok((small_bm, small_bn, true));
        }
        return Ok((bm, bn, false));
    }
    if let Some((bm, bn)) = mqa_fwd_block_config_small(head_dim)
        && compute_smem(bm, bn, head_dim, elem_bytes) <= max_smem
    {
        return Ok((bm, bn, true));
    }

    let reason = match mqa_fwd_block_config_small(head_dim) {
        Some((bm, bn)) => format!(
            "MQA/GQA forward for head_dim={} needs {} bytes of shared memory \
             (smallest block config BLOCK_M={}, BLOCK_N={}, {}-byte elements) but this GPU \
             allows at most {} bytes per block",
            head_dim,
            compute_smem(bm, bn, head_dim, elem_bytes),
            bm,
            bn,
            elem_bytes,
            max_smem
        ),
        None => format!(
            "MQA/GQA kernels support head_dim 32/64/128, got {}",
            head_dim
        ),
    };
    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason,
    })
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
/// [`mqa_fwd_block_config`] is NOT usable here: the forward stages in the tensor
/// dtype with `head_dim + 1` padding, while the backward always stages 4 tiles of
/// f32 with no padding, which is larger at every supported head_dim.
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
///   See `mqa_fwd_block_config_large` / `mqa_fwd_block_config_small` in
///   this file, which mirror those instantiations. Any other head_dim has no
///   kernel symbol to call.
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
/// Gates PREFILL only: both call sites in `flash.rs` route `seq_len_q == 1` to
/// the decode path before reaching this check.
pub fn should_use_mqa_gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> bool {
    if num_kv_heads == 0 {
        return false;
    }
    num_heads.is_multiple_of(num_kv_heads) && matches!(head_dim, 32 | 64 | 128)
}

#[cfg(test)]
mod mqa_fwd_block_config_tests {
    use super::*;
    use numr::runtime::cuda::is_cuda_available;

    // elem_bytes=4 (F32) for all cases below. Every supported head_dim (32,
    // 64, 128) has both a large and a small config here, unlike the flash
    // path, so there is no "no small config" case for this kernel family.

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
        let (large_bm, large_bn) =
            mqa_fwd_block_config_large(32).expect("head_dim 32 has a large config");
        let (small_bm, small_bn) =
            mqa_fwd_block_config_small(32).expect("head_dim 32 has a small config");
        let max_smem = device_max_smem();
        if compute_smem(large_bm, large_bn, 32, 4) > max_smem
            || compute_smem(small_bm, small_bn, 32, 4) > max_smem
        {
            eprintln!("device shared memory too small for this precondition, skipping");
            return;
        }
        let (block_m, _block_n, use_sm_kernel) = mqa_fwd_block_config(32, 4, 2).unwrap();
        assert!(use_sm_kernel);
        assert_eq!(block_m, small_bm);
    }

    #[test]
    fn long_seq_len_q_keeps_large_config() {
        if !require_cuda() {
            return;
        }
        let (large_bm, large_bn) =
            mqa_fwd_block_config_large(32).expect("head_dim 32 has a large config");
        if compute_smem(large_bm, large_bn, 32, 4) > device_max_smem() {
            eprintln!("device shared memory too small for this precondition, skipping");
            return;
        }
        let (block_m, _block_n, use_sm_kernel) = mqa_fwd_block_config(32, 4, 4096).unwrap();
        assert!(!use_sm_kernel);
        assert_eq!(block_m, large_bm);
    }

    #[test]
    fn smem_forcing_to_small_config_is_unchanged_by_seq_len_q() {
        if !require_cuda() {
            return;
        }
        // head_dim=128 large needs ~129KB, over the opt-in limit on many GPUs
        // (e.g. consumer Ampere's ~100KB) even where it fits on others — so
        // this precondition, not the seq_len_q heuristic, is what is under
        // test: capability forcing must win regardless of seq_len_q.
        let (large_bm, large_bn) =
            mqa_fwd_block_config_large(128).expect("head_dim 128 has a large config");
        let (small_bm, small_bn) =
            mqa_fwd_block_config_small(128).expect("head_dim 128 has a small config");
        let max_smem = device_max_smem();
        if compute_smem(large_bm, large_bn, 128, 4) <= max_smem {
            eprintln!("device shared memory fits the large config here, skipping");
            return;
        }
        if compute_smem(small_bm, small_bn, 128, 4) > max_smem {
            eprintln!("device shared memory too small for the small config too, skipping");
            return;
        }
        // A long seq_len_q would normally keep the large config, but it does
        // not fit here, so the small config is forced either way.
        let (block_m, _block_n, use_sm_kernel) = mqa_fwd_block_config(128, 4, 4096).unwrap();
        assert!(use_sm_kernel);
        assert_eq!(block_m, small_bm);
    }

    #[test]
    fn unsupported_head_dim_still_errors() {
        // No config exists for head_dim=999 at any seq_len_q, so this must
        // error regardless of the device's shared-memory limit.
        assert!(mqa_fwd_block_config(999, 4, 4096).is_err());
    }
}
