//! Paged attention BACKWARD tile/block-size selection and shared-memory sizing.
//!
//! Split out of `paged_attention.rs` to keep that file's `PagedAttentionOps`
//! trait impl as wiring only and this file under the crate's `cuda/*.rs` size
//! limit — mirrors `mqa_gqa/block_config.rs`'s split from
//! `mqa_gqa/fwd.rs`/`mqa_gqa/bwd.rs`. The forward counterpart is
//! `paged_attention_fwd_block_config.rs`, split separately (rather than
//! combined into one fwd+bwd file, as `mqa_gqa/block_config.rs` does) because
//! the combined fwd+bwd content here exceeded the size limit on its own.
//! Consumed by `paged_attention_bwd.rs`'s kernel launcher.

use crate::error::{Error, Result};
use numr::dtype::DType;
use std::env;

use super::flash_utils::device_max_smem;

/// Small-memory block config for paged attention backward (`_small` kernels).
/// Sized to fit in 48KB shared memory. See [`bwd_smem_size`] for the layout.
fn bwd_block_config_small(head_dim: usize, dtype: DType) -> Result<(usize, usize)> {
    match (dtype, head_dim) {
        (DType::F32, 64) => Ok((32, 32)),  // (96+64)*64*4 = 40KB
        (DType::F32, 128) => Ok((16, 16)), // (48+32)*128*4 = 40KB
        (DType::F16 | DType::BF16, 64) => Ok((64, 32)), // (192+64)*64*2 = 32KB
        (DType::F16 | DType::BF16, 128) => Ok((32, 32)), // (96+64)*128*2 = 40KB
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for paged attention backward",
                head_dim
            ),
        }),
    }
}

/// Large-tile block config of the unsuffixed `paged_flash_attention_bwd_{head_dim}_{dtype}`
/// kernels (`BLOCK_M=128, BLOCK_N=64`). Must stay in sync with the `extern "C"`
/// instantiations in `paged_attention_bwd.cu`. There is no FP8 backward kernel at all.
fn bwd_block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        64 | 128 => Some((128, 64)),
        _ => None,
    }
}

/// Shared memory bytes the paged attention BACKWARD kernel needs.
///
/// Layout in `paged_attention_bwd.cu`: `[K: BLOCK_N][V: BLOCK_N][dO: BLOCK_N..]`
/// — concretely `K_smem = smem + BLOCK_M*HD`, `V_smem = smem + BLOCK_M*HD + BLOCK_N*HD`,
/// `dO_smem = smem + BLOCK_M*HD + 2*BLOCK_N*HD`, `O_smem = smem + 2*BLOCK_M*HD + 2*BLOCK_N*HD`,
/// with Q implicitly at the base (`smem`) and O's tile bringing the total to
/// `3*BLOCK_M + 2*BLOCK_N` rows of `HEAD_DIM` elements, no bank-conflict padding.
///
/// This is the DYNAMIC allocation only. Each kernel also declares two static
/// shared arrays — `D_smem` and `lse_smem`, `BLOCK_M` floats each — so the
/// block's real shared-memory footprint is this value plus `2*BLOCK_M*4` bytes.
/// That term is at most 1KB (large tile, `BLOCK_M=128`) and no config sits
/// within 1KB of a device opt-in limit, so it does not change any verdict here.
pub(super) fn bwd_smem_size(
    block_m: usize,
    block_n: usize,
    head_dim: usize,
    elem_bytes: usize,
) -> usize {
    (3 * block_m + 2 * block_n) * head_dim * elem_bytes
}

/// Escape hatch for the backward large tile. `bwd_block_config` is SMALL-ONLY
/// by default (see its doc comment) because the large bwd tile is unmeasured;
/// set `BOOSTR_PAGED_BWD_TILE=large` to force it for measurement on real
/// hardware. Any other value (unset, `small`, `auto`, unrecognized) keeps the
/// small-only default. Mirrors `paged_attention_fwd_block_config`'s forward
/// escape hatch (`prefill_tile_override`), minus the small-side override —
/// there is nothing to override *to* small when small is already the default.
fn bwd_tile_override_large() -> bool {
    env::var("BOOSTR_PAGED_BWD_TILE")
        .map(|v| v.eq_ignore_ascii_case("large"))
        .unwrap_or(false)
}

/// Pick the paged attention backward tile that fits this device's opt-in shared
/// memory. Returns `(block_m, block_n, use_large)`. No FP8 backward kernel exists.
///
/// SMALL-ONLY by default: unlike the forward prefill policy, the backward
/// large tile has never been measured on real hardware, so it is not
/// selected automatically. `bwd_block_config_large` / `bwd_smem_size` / the
/// capability-gate plumbing (`set_smem_attribute` sizing the large kernel's
/// opt-in shared memory) stay reachable through `BOOSTR_PAGED_BWD_TILE=large`
/// — see [`bwd_tile_override_large`] — so the large bwd path stays compiled
/// and honest rather than silently dead code.
///
/// At head_dim=128, the F32 large config needs 256KB, beyond any current
/// device's opt-in shared-memory limit regardless — the capability gate
/// below falls back to `small` for it even under the override, no
/// special-casing required.
pub(super) fn bwd_block_config(head_dim: usize, dtype: DType) -> Result<(usize, usize, bool)> {
    let elem_bytes = dtype.size_in_bytes();
    let max_smem = device_max_smem();

    if bwd_tile_override_large()
        && let Some((bm, bn)) = bwd_block_config_large(head_dim)
    {
        let smem = bwd_smem_size(bm, bn, head_dim, elem_bytes);
        if smem <= max_smem {
            return Ok((bm, bn, true));
        }
        eprintln!(
            "BOOSTR_PAGED_BWD_TILE=large requested but paged attention backward for \
             head_dim={} needs shared memory exceeding this device's {} byte opt-in \
             limit; falling back to the small tile",
            head_dim, max_smem
        );
    }

    let (bm, bn) = bwd_block_config_small(head_dim, dtype)?;
    let smem = bwd_smem_size(bm, bn, head_dim, elem_bytes);
    if smem <= max_smem {
        return Ok((bm, bn, false));
    }

    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason: format!(
            "unsupported head_dim={} for paged attention backward on this GPU \
             (max shared memory: {} bytes)",
            head_dim, max_smem
        ),
    })
}
