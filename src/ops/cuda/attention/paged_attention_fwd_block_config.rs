//! Paged attention PREFILL FORWARD tile/block-size selection and
//! shared-memory sizing.
//!
//! Split out of `paged_attention.rs` to keep that file's `PagedAttentionOps`
//! trait impl as wiring only and this file under the crate's `cuda/*.rs` size
//! limit — mirrors `mqa_gqa/block_config.rs`'s split from
//! `mqa_gqa/fwd.rs`/`mqa_gqa/bwd.rs`. The backward counterpart is
//! `paged_attention_bwd_block_config.rs`, split out separately once the
//! combined fwd+bwd file itself exceeded the limit. Consumed by
//! `paged_attention_fwd.rs`'s kernel launcher.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::CudaDevice;
use std::env;

use super::flash_utils::device_max_smem;

/// Small-memory block config for paged attention forward (`_small` kernels).
/// Sized to fit in 48KB shared memory. See [`fwd_smem_size`] for the layout.
fn fwd_block_config_small(head_dim: usize, dtype: DType) -> Result<(usize, usize)> {
    match (dtype, head_dim) {
        // FP32: 4 bytes per element
        (DType::F32, 64) => Ok((64, 32)),  // (64+32+32)*64*4 = 32KB
        (DType::F32, 128) => Ok((32, 32)), // (32+32+32)*128*4 = 48KB
        // FP16/BF16: 2 bytes per element
        (DType::F16 | DType::BF16, 64) => Ok((64, 32)), // (64+32+32)*64*2 = 16KB
        (DType::F16 | DType::BF16, 128) => Ok((32, 32)), // (32+32+32)*128*2 = 24KB
        // FP8: uses FP32 smem for compute
        (DType::FP8E4M3 | DType::FP8E5M2, 64) => Ok((64, 32)), // (64+64)*64*4 = 32KB
        (DType::FP8E4M3 | DType::FP8E5M2, 128) => Ok((32, 32)), // (32+64)*128*4 = 48KB
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for paged attention. Supported: 64, 128",
                head_dim
            ),
        }),
    }
}

/// Large-tile block config of the unsuffixed `paged_flash_attention_fwd_{head_dim}_{dtype}`
/// kernels (`BLOCK_M=128, BLOCK_N=64`). Must stay in sync with the `extern "C"`
/// instantiations in `paged_attention.cu`. FP8 has no large kernel — never queried for it.
fn fwd_block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        64 | 128 => Some((128, 64)),
        _ => None,
    }
}

/// Shared memory bytes the paged attention FORWARD kernel needs.
///
/// Layout in `paged_attention.cu` (`paged_flash_attention_fwd_*_impl`):
/// `[Q: BLOCK_M x HEAD_DIM][K: BLOCK_N x HEAD_DIM][V: BLOCK_N x HEAD_DIM]`
/// (`Q_smem_flat = smem`, `K_smem_flat = smem + BLOCK_M*HEAD_DIM`,
/// `V_smem_flat = smem + BLOCK_M*HEAD_DIM + BLOCK_N*HEAD_DIM`), with NO
/// bank-conflict padding — unlike flash/varlen's `HEAD_DIM + 1` layout, so
/// `flash_utils::compute_smem` must NOT be reused here.
pub(super) fn fwd_smem_size(
    block_m: usize,
    block_n: usize,
    head_dim: usize,
    elem_bytes: usize,
) -> usize {
    (block_m + 2 * block_n) * head_dim * elem_bytes
}

/// Profiling/diagnostic escape hatch: force the paged attention PREFILL forward
/// tile choice for A/B measurement on the same device. Read once per call from
/// `BOOSTR_PAGED_PREFILL_TILE`. Any value other than `large`/`small`
/// (unset, empty, `auto`, or unrecognized) is the normal capability-gated
/// selection. `large` is never honoured when the device cannot fit it — see
/// [`fwd_block_config`], which falls back to `small` and reports the fallback.
fn prefill_tile_override() -> Option<bool> {
    match env::var("BOOSTR_PAGED_PREFILL_TILE") {
        Ok(v) if v.eq_ignore_ascii_case("large") => Some(true),
        Ok(v) if v.eq_ignore_ascii_case("small") => Some(false),
        _ => None,
    }
}

/// Resident blocks per compute unit the prefill large-tile grid-coverage
/// check aims for: a single wave. The large tile's per-row cost is only
/// repaid once its (narrower, because `block_m` is bigger) grid still puts a
/// block on every compute unit; short of that, occupancy collapse dominates.
/// This is a different cost model from `decode_split::DECODE_BLOCKS_PER_UNIT`
/// (8, filling several resident blocks per unit for a memory-latency-bound
/// kernel) — one wave is the whole target here, not a floor to build on.
const PAGED_PREFILL_BLOCKS_PER_UNIT: usize = 1;

/// The prefill forward large-tile performance policy, layered on top of the
/// capability gate (the large tile must already fit shared memory — checked
/// by the caller before this runs). Only F16/BF16 ever prefer the large
/// tile; F32 and FP8 never do.
///
/// - `head_dim == 64`: the small tile's own `block_m` is `small_block_m`.
///   Enlarging `block_m` to the large tile's value only changes the grid
///   width once `seq_len_q` needs more than one small-tile row, so below
///   that the large tile's halved K-loop trip count is free and always
///   wins.
/// - `head_dim == 128`: the small tile's `block_m` is a quarter of the
///   large tile's, so the large tile's extra per-row work is real and must
///   be repaid — only once the large tile's grid
///   (`num_heads * batch_size * ceil(seq_len_q / large_block_m)`) still
///   covers the device (`compute_units`, one wave) does the halved K-loop
///   win outright.
///
/// `compute_units == 0` (unknown device profile) never selects large
/// through the grid-coverage branch, matching the conservative default in
/// `decode_split::decode_split_for_units`.
#[allow(clippy::too_many_arguments)]
fn fwd_prefer_large(
    head_dim: usize,
    dtype: DType,
    seq_len_q: usize,
    num_heads: usize,
    batch_size: usize,
    small_block_m: usize,
    compute_units: usize,
) -> bool {
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return false;
    }
    match head_dim {
        64 => seq_len_q <= small_block_m,
        128 => {
            if compute_units == 0 {
                return false;
            }
            let large_blocks = num_heads * batch_size * seq_len_q.div_ceil(128);
            large_blocks >= compute_units * PAGED_PREFILL_BLOCKS_PER_UNIT
        }
        _ => false,
    }
}

/// Pick the paged attention PREFILL forward tile that fits this device's opt-in
/// shared memory. Returns `(block_m, block_n, use_large)`; `use_large` selects
/// between the unsuffixed large kernel and the `_small` kernel.
///
/// FP8 has only `_small` kernels compiled (no large FP8 kernel exists), so FP8
/// always returns the small config, unconditionally and regardless of
/// `BOOSTR_PAGED_PREFILL_TILE`.
///
/// Reads `BOOSTR_PAGED_PREFILL_TILE` for the override; see
/// [`fwd_block_config_with_override`] for the same selection with the
/// override passed explicitly instead (used by tests, which cannot safely
/// set a process-wide env var from parallel test threads).
#[allow(clippy::too_many_arguments)]
pub(super) fn fwd_block_config(
    head_dim: usize,
    dtype: DType,
    seq_len_q: usize,
    num_heads: usize,
    batch_size: usize,
    device_index: usize,
) -> Result<(usize, usize, bool)> {
    fwd_block_config_with_override(
        head_dim,
        dtype,
        seq_len_q,
        num_heads,
        batch_size,
        device_index,
        prefill_tile_override(),
    )
}

/// Core of [`fwd_block_config`]: capability gate first (large must fit this
/// device's opt-in shared memory), then the measured performance policy
/// ([`fwd_prefer_large`]), then `override_large` — `Some(true)`/`Some(false)`
/// pins one side (still refused, with a fallback, if `Some(true)` can't fit);
/// `None` uses the policy.
#[allow(clippy::too_many_arguments)]
pub(super) fn fwd_block_config_with_override(
    head_dim: usize,
    dtype: DType,
    seq_len_q: usize,
    num_heads: usize,
    batch_size: usize,
    device_index: usize,
    override_large: Option<bool>,
) -> Result<(usize, usize, bool)> {
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        let (bm, bn) = fwd_block_config_small(head_dim, dtype)?;
        return Ok((bm, bn, false));
    }

    let elem_bytes = dtype.size_in_bytes();
    let max_smem = device_max_smem();
    let (small_bm, small_bn) = fwd_block_config_small(head_dim, dtype)?;

    // Capability gate: does the large tile fit this device's opt-in shared
    // memory at all?
    let large_fits = fwd_block_config_large(head_dim).and_then(|(bm, bn)| {
        let smem = fwd_smem_size(bm, bn, head_dim, elem_bytes);
        (smem <= max_smem).then_some((bm, bn))
    });

    if let Some((bm, bn)) = large_fits {
        let want_large = match override_large {
            Some(forced) => forced,
            None => {
                let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
                fwd_prefer_large(
                    head_dim,
                    dtype,
                    seq_len_q,
                    num_heads,
                    batch_size,
                    small_bm,
                    compute_units,
                )
            }
        };
        if want_large {
            return Ok((bm, bn, true));
        }
    } else if override_large == Some(true) {
        eprintln!(
            "BOOSTR_PAGED_PREFILL_TILE=large requested but paged attention prefill \
             forward for head_dim={} needs shared memory exceeding this device's {} byte \
             opt-in limit; falling back to the small tile",
            head_dim, max_smem
        );
    }

    let smem = fwd_smem_size(small_bm, small_bn, head_dim, elem_bytes);
    if smem <= max_smem {
        return Ok((small_bm, small_bn, false));
    }

    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason: format!(
            "unsupported head_dim={} for paged attention prefill forward on this GPU \
             (max shared memory: {} bytes). Supported: 64, 128",
            head_dim, max_smem
        ),
    })
}

/// Test-only entry point exposing [`fwd_block_config_with_override`]'s
/// decision together with the shared-memory bytes it computed and this
/// device's opt-in limit, so parity tests can assert the capability gate
/// never returns a tile that does not fit — without reaching into the
/// crate-private `fwd_smem_size`/`device_max_smem` helpers themselves.
/// Returns `(block_m, block_n, use_large, smem_bytes, device_max_smem_bytes)`.
#[doc(hidden)]
pub fn fwd_prefill_tile_for_test(
    head_dim: usize,
    dtype: DType,
    seq_len_q: usize,
    num_heads: usize,
    batch_size: usize,
    device_index: usize,
    override_large: Option<bool>,
) -> Result<(usize, usize, bool, usize, usize)> {
    let (block_m, block_n, use_large) = fwd_block_config_with_override(
        head_dim,
        dtype,
        seq_len_q,
        num_heads,
        batch_size,
        device_index,
        override_large,
    )?;
    // FP8 computes shared memory in FP32 regardless of the tensor's own
    // element size — see `fwd_smem_size`'s FP8 caller in
    // `paged_attention_fwd_fp8_impl`.
    let elem_bytes = if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        4
    } else {
        dtype.size_in_bytes()
    };
    let smem = fwd_smem_size(block_m, block_n, head_dim, elem_bytes);
    Ok((block_m, block_n, use_large, smem, device_max_smem()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_never_prefers_large() {
        assert!(!fwd_prefer_large(64, DType::F32, 8, 32, 1, 64, 100));
        assert!(!fwd_prefer_large(128, DType::F32, 2048, 32, 1, 32, 100));
    }

    #[test]
    fn fp8_never_prefers_large() {
        assert!(!fwd_prefer_large(64, DType::FP8E4M3, 8, 32, 1, 64, 100));
        assert!(!fwd_prefer_large(128, DType::FP8E5M2, 2048, 32, 1, 32, 100));
    }

    #[test]
    fn head_dim_64_prefers_large_only_within_the_small_tile_row() {
        // small_block_m = 64 at head_dim=64: below/at it the large tile
        // produces the same grid width, so it is free.
        assert!(fwd_prefer_large(64, DType::F16, 8, 32, 1, 64, 100));
        assert!(fwd_prefer_large(64, DType::BF16, 64, 32, 1, 64, 100));
        // Past it, the large tile's BLOCK_M=128 starts shrinking the grid.
        assert!(!fwd_prefer_large(64, DType::F16, 65, 32, 1, 64, 100));
        assert!(!fwd_prefer_large(64, DType::BF16, 2048, 32, 1, 64, 100));
    }

    #[test]
    fn head_dim_128_prefers_large_only_once_the_grid_covers_the_device() {
        let compute_units = 80;
        // num_heads=32, batch=1: ceil(q/128) blocks per (head,batch) pair.
        // At q=32 the grid is 32 blocks, short of 80 compute units.
        assert!(!fwd_prefer_large(
            128,
            DType::F16,
            32,
            32,
            1,
            32,
            compute_units
        ));
        // At q=1024 the grid is 32*8=256 blocks, comfortably >= 80.
        assert!(fwd_prefer_large(
            128,
            DType::BF16,
            1024,
            32,
            1,
            32,
            compute_units
        ));
    }

    #[test]
    fn unknown_compute_unit_profile_never_prefers_large_at_head_dim_128() {
        assert!(!fwd_prefer_large(128, DType::F16, 1 << 20, 32, 1, 32, 0));
    }

    #[test]
    fn override_beats_policy_in_fwd_block_config_with_override() {
        // head_dim=64, F32 never prefers large by policy, but an explicit
        // override still forces it when the capability gate allows it.
        let forced_large = fwd_block_config_with_override(64, DType::F32, 8, 32, 1, 0, Some(true));
        // On a CPU-only build (or a CUDA build with a suitable device) the
        // large F32 head_dim=64 tile fits comfortably (32KB); this asserts
        // the override wins over the policy whenever the gate passes, not a
        // specific device outcome.
        if let Ok((_, _, use_large)) = forced_large {
            assert!(use_large || device_max_smem() == 0);
        }
    }
}
