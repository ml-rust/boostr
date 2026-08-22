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
/// `mqa_gqa_bwd.cu` declares `extern __shared__ float smem[]` in every backward
/// impl (`mqa_gqa_bwd_fp32_impl`, `mqa_gqa_bwd_fp16_impl`, `mqa_gqa_bwd_dtype_impl`)
/// and stages K/V/Q/dO as f32 there, converting on load. The requirement is
/// therefore independent of the tensor dtype.
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
pub(super) fn mqa_fwd_block_config(
    head_dim: usize,
    elem_bytes: usize,
) -> Result<(usize, usize, bool)> {
    let max_smem = device_max_smem();

    if let Some((bm, bn)) = mqa_fwd_block_config_large(head_dim)
        && compute_smem(bm, bn, head_dim, elem_bytes) <= max_smem
    {
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

/// Returns true if the GQA ratio warrants using dedicated MQA/GQA kernels.
///
/// Heuristic: use dedicated kernels when:
/// - num_heads / num_kv_heads >= 4 (extreme GQA ratio)
/// - head_dim is 32, 64, or 128 (supported by MQA/GQA kernels)
pub fn should_use_mqa_gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> bool {
    if num_kv_heads == 0 {
        return false;
    }
    let ratio = num_heads / num_kv_heads;
    ratio >= 4 && matches!(head_dim, 32 | 64 | 128)
}
