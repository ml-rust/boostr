//! VarLen attention tile/block-size selection and shared-memory sizing.
//!
//! Split out of `varlen_attention.rs` to keep that file's `VarLenAttentionOps`
//! trait impl as wiring only and this file under the crate's `cuda/*.rs` size
//! limit — mirrors `paged_attention_bwd_block_config.rs`'s split from
//! `paged_attention_bwd.rs`.

use crate::error::{Error, Result};
use numr::dtype::DType;

use super::flash_utils::{compute_smem, device_max_smem};

/// Which compiled kernel variant a `(head_dim, dtype)` request resolves to.
/// The variant, not a bare `bool`, is what determines the kernel-name suffix
/// — this is the fix for a real regression: head_dim=256 has exactly one
/// compiled config, exposed ONLY under the unsuffixed name
/// (`varlen_flash_attention_{fwd,bwd}_256_{fp32,fp16}`), but a selector that
/// reported only "did the large tile fit" treated 256 as "large didn't fit"
/// and appended `_small`, producing a symbol
/// (`varlen_flash_attention_fwd_256_fp32_small`) that was never compiled —
/// `CUDA_ERROR_NOT_FOUND`. `Base256` exists so head_dim=256 can never be
/// confused with `Small`, and the suffix is derived from the variant in one
/// place ([`TileVariant::suffix`]), never re-decided at the kernel-name
/// call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TileVariant {
    /// Unsuffixed large tile (`BLOCK_M=128, BLOCK_N=64`). head_dim 64/128 only.
    Large,
    /// `_small`-suffixed fallback tile. head_dim 64 (F32 only) / 128.
    Small,
    /// Unsuffixed kernel that is head_dim=256's ONLY compiled config
    /// (`<256,16,16>` F32, `<256,32,32>` F16). Distinct from `Large` even
    /// though both resolve to an empty suffix: head_dim=256 has no `_small`
    /// sibling to fall back to, so it must never be reached through the
    /// large/small capability-gate branch that can produce `Small`.
    Base256,
}

impl TileVariant {
    /// Kernel-name suffix for this variant — the ONLY place that maps a
    /// variant to a suffix, so a resolved tile and the symbol name loaded for
    /// it cannot drift apart.
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            TileVariant::Large | TileVariant::Base256 => "",
            TileVariant::Small => "_small",
        }
    }
}

/// Large-tile ("full") config: `(BLOCK_M, BLOCK_N) = (128, 64)`, the proven
/// tile used by the unsuffixed `varlen_flash_attention_{fwd,bwd}_{head_dim}_{dtype}`
/// kernels. `None` for any head_dim other than 64/128 — head_dim=256 is
/// handled entirely separately (see [`head_dim_256_config`]), never through
/// this function.
fn block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        64 | 128 => Some((128, 64)),
        _ => None,
    }
}

/// Small-tile fallback config of the `varlen_flash_attention_{fwd,bwd}_{head_dim}_{dtype}_small`
/// kernels. `None` when no `_small` kernel is instantiated for this
/// `(head_dim, dtype)` pair — head_dim=64/F16 fits the large tile on every
/// opt-in shared-memory limit this crate targets (49920 B backward, under a
/// 99KB device), so no fallback was compiled for it. Never called for
/// head_dim=256 — see [`head_dim_256_config`].
fn block_config_small(head_dim: usize, dtype: DType) -> Option<(usize, usize)> {
    match (head_dim, dtype) {
        (64, DType::F32) => Some((32, 32)),
        (128, DType::F32) => Some((16, 16)),
        (128, DType::F16) => Some((32, 32)),
        _ => None,
    }
}

/// head_dim=256's single compiled config — `<256,16,16>` F32, `<256,32,32>`
/// F16 — exposed ONLY under the unsuffixed `varlen_flash_attention_{fwd,bwd}_256_{fp32,fp16}`
/// name (`TileVariant::Base256`). There is no `_small` sibling; adding one
/// would just duplicate this same kernel under a second symbol name.
fn head_dim_256_config(dtype: DType) -> (usize, usize) {
    match dtype {
        DType::F16 => (32, 32),
        _ => (16, 16), // F32 and any future dtype default to 16/16
    }
}

/// Shared memory bytes the varlen attention BACKWARD kernel needs.
///
/// Unlike `flash_utils::compute_bwd_smem` (the `flash_v2_bwd.cu` layout, which
/// has no bank-conflict padding), varlen's backward kernels DO pad each row to
/// `HEAD_DIM + 1` — same padding convention as the forward layout in
/// `flash_utils::compute_smem`, which this file reuses directly for forward
/// sizing. Layout: `[Q][K][V][dO]`, each `BLOCK_M` or `BLOCK_N` rows of
/// `HEAD_DIM + 1` elements — see `varlen_flash_attention_bwd_fp32_impl` /
/// `_fp16_impl` in `varlen_attention_bwd.cu` / `varlen_attention_bwd_fp16.cu`.
fn varlen_bwd_smem(block_m: usize, block_n: usize, head_dim: usize, elem_bytes: usize) -> usize {
    let head_stride = head_dim + 1;
    (2 * block_m + 2 * block_n) * head_stride * elem_bytes
}

/// Pick the varlen attention tile that fits this device's opt-in shared
/// memory. Returns `(block_m, block_n, variant)`. `varlen_attention.rs`'s
/// production path calls this with `override_large: None`; the
/// `_with_tile_for_test` entry points in `varlen_attention_fwd.rs` /
/// `varlen_attention_bwd.rs` pass `Some(bool)` to force one side for parity
/// tests. `override_large` has no effect on head_dim=256, which always
/// resolves to `TileVariant::Base256` — see below.
///
/// Sized on the BACKWARD requirement, not the forward one: `varlen_bwd_smem`
/// is strictly larger than `compute_smem` at any given tile (`2*BLOCK_M` vs
/// `BLOCK_M` rows of Q/dO), so fwd and bwd need the same decision to stay
/// consistent for a given call site — both `varlen_attention_fwd_impl_inner`
/// and `varlen_attention_bwd_impl_inner` call this same function. This means
/// a forward-only call may take the small tile even in cases where the large
/// tile would have fit forward alone; that's the trade-off for a single tile
/// choice per `(head_dim, dtype)`.
///
/// | head_dim | dtype | Large (128,64)   | Small           | Base256   |
/// |----------|-------|------------------|-----------------|-----------|
/// | 64       | F32   | candidate        | 32,32 fallback  | n/a       |
/// | 64       | F16   | candidate        | none — n/a      | n/a       |
/// | 128      | F32   | candidate        | 16,16 fallback  | n/a       |
/// | 128      | F16   | candidate        | 32,32 fallback  | n/a       |
/// | 256      | F32   | n/a              | n/a             | 16,16     |
/// | 256      | F16   | n/a              | n/a             | 32,32     |
///
/// head_dim=256 is handled first and unconditionally: it has exactly one
/// compiled config, so there is no large/small choice to make and no
/// `override_large` to honor — `Some(_)` errors rather than silently
/// ignoring the caller's request.
///
/// For head_dim 64/128: capability gate first (does the requested tile fit
/// this device's opt-in shared memory, sized on the backward requirement?),
/// then `override_large` picks a side. `Some(true)`/`Some(false)` pins one
/// tile — still refused if that tile does not exist or does not fit this
/// device — `None` tries large first and falls back to small.
///
/// `pub(crate)`, not `pub(super)`, so the `_with_tile_for_test` entry points
/// in `varlen_attention_fwd.rs` / `varlen_attention_bwd.rs` can call it with
/// an explicit override, which `tests/varlen_tile_fallback_cuda.rs` uses to
/// drive both sides of the large/small parity comparison — rather than a
/// process-wide env var (Rust tests run multi-threaded in one process, so
/// `std::env::set_var` from a test would race).
pub(crate) fn block_config_with_override(
    head_dim: usize,
    dtype: DType,
    override_large: Option<bool>,
) -> Result<(usize, usize, TileVariant)> {
    let max_smem = device_max_smem();
    let elem_bytes = dtype.size_in_bytes();

    if head_dim == 256 {
        if override_large.is_some() {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: "varlen attention: head_dim=256 has a single compiled tile \
                         config (no large/small pair), so no tile override applies"
                    .to_string(),
            });
        }
        let (bm, bn) = head_dim_256_config(dtype);
        let smem = varlen_bwd_smem(bm, bn, head_dim, elem_bytes);
        if smem > max_smem {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "varlen attention: head_dim=256 dtype={:?} needs {} bytes, exceeding \
                     this device's {}KB opt-in shared memory limit",
                    dtype,
                    smem,
                    max_smem / 1024
                ),
            });
        }
        return Ok((bm, bn, TileVariant::Base256));
    }

    match override_large {
        Some(true) => {
            let (bm, bn) = block_config_large(head_dim).ok_or_else(|| Error::InvalidArgument {
                arg: "head_dim",
                reason: format!("varlen attention: no large-tile kernel for head_dim={head_dim}"),
            })?;
            let smem = varlen_bwd_smem(bm, bn, head_dim, elem_bytes);
            if smem > max_smem {
                return Err(Error::InvalidArgument {
                    arg: "head_dim",
                    reason: format!(
                        "varlen attention: large tile for head_dim={} dtype={:?} needs {} \
                         bytes, exceeding this device's {}KB opt-in shared memory limit",
                        head_dim,
                        dtype,
                        smem,
                        max_smem / 1024
                    ),
                });
            }
            Ok((bm, bn, TileVariant::Large))
        }
        Some(false) => {
            let (bm, bn) =
                block_config_small(head_dim, dtype).ok_or_else(|| Error::InvalidArgument {
                    arg: "head_dim",
                    reason: format!(
                        "varlen attention: no small-tile kernel for head_dim={head_dim} \
                         dtype={dtype:?}"
                    ),
                })?;
            let smem = varlen_bwd_smem(bm, bn, head_dim, elem_bytes);
            if smem > max_smem {
                return Err(Error::InvalidArgument {
                    arg: "head_dim",
                    reason: format!(
                        "varlen attention: small tile for head_dim={} dtype={:?} needs {} \
                         bytes, exceeding this device's {}KB opt-in shared memory limit",
                        head_dim,
                        dtype,
                        smem,
                        max_smem / 1024
                    ),
                });
            }
            Ok((bm, bn, TileVariant::Small))
        }
        None => {
            if let Some((bm, bn)) = block_config_large(head_dim)
                && varlen_bwd_smem(bm, bn, head_dim, elem_bytes) <= max_smem
            {
                return Ok((bm, bn, TileVariant::Large));
            }

            if let Some((bm, bn)) = block_config_small(head_dim, dtype)
                && varlen_bwd_smem(bm, bn, head_dim, elem_bytes) <= max_smem
            {
                return Ok((bm, bn, TileVariant::Small));
            }

            Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "varlen attention: no tile configuration for head_dim={} dtype={:?} fits \
                     this GPU (max shared memory: {}KB)",
                    head_dim,
                    dtype,
                    max_smem / 1024
                ),
            })
        }
    }
}

/// Forward shared memory bytes for a chosen tile — reuses `flash_utils::compute_smem`,
/// whose padded `(BLOCK_M + 2*BLOCK_N) * (HEAD_DIM+1) * elem_bytes` formula is
/// identical to varlen's forward smem layout.
pub(super) fn fwd_smem_size(
    block_m: usize,
    block_n: usize,
    head_dim: usize,
    elem_bytes: usize,
) -> usize {
    compute_smem(block_m, block_n, head_dim, elem_bytes)
}

/// Backward shared memory bytes for a chosen tile. See [`varlen_bwd_smem`] for
/// why this is a local formula rather than `flash_utils::compute_bwd_smem`.
pub(super) fn bwd_smem_size(
    block_m: usize,
    block_n: usize,
    head_dim: usize,
    elem_bytes: usize,
) -> usize {
    varlen_bwd_smem(block_m, block_n, head_dim, elem_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cuda::is_cuda_available;

    /// Same gate the CUDA integration tests use: the `cuda` feature can be on
    /// while no device is present (`device_max_smem` reads a live CUDA
    /// context), so the suite must skip, not crash.
    fn require_cuda() -> bool {
        if !is_cuda_available() {
            eprintln!("CUDA feature enabled but runtime unavailable, skipping");
            return false;
        }
        true
    }

    #[test]
    fn head_dim_256_never_produces_a_small_suffix() {
        if !require_cuda() {
            return;
        }
        for dtype in [DType::F32, DType::F16] {
            let (_, _, variant) = block_config_with_override(256, dtype, None).expect(
                "head_dim=256 must resolve without a device-dependent capability gate \
                         failure for a reasonably sized opt-in shared-memory limit",
            );
            assert_eq!(variant.suffix(), "");
            assert_ne!(variant, TileVariant::Small);
        }
    }

    #[test]
    fn head_dim_256_rejects_a_tile_override() {
        if !require_cuda() {
            return;
        }
        assert!(block_config_with_override(256, DType::F32, Some(true)).is_err());
        assert!(block_config_with_override(256, DType::F32, Some(false)).is_err());
    }

    #[test]
    fn head_dim_64_fp16_has_no_small_fallback() {
        assert!(block_config_small(64, DType::F16).is_none());
    }
}
