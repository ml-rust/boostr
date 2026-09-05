//! MLA SDPA tile/block-size selection and shared-memory sizing.
//!
//! Split out of `mla.rs` to keep that file's `MlaOps` trait impl as wiring
//! only — mirrors `varlen_attention_block_config.rs`'s split from
//! `varlen_attention.rs` and `paged_attention_fwd_block_config.rs`'s from
//! `paged_attention_fwd.rs`.
//!
//! `sdpa.cu` instantiates one templated `sdpa_impl<T, BLOCK_M, BLOCK_N>` at
//! two tiles per dtype; this module is what decides which of the two a shape
//! gets on this device.

use crate::error::{Error, Result};
use numr::dtype::DType;

use super::flash::flash_utils::device_max_smem;

/// Length of the per-thread `float O_local[SDPA_MAX_HEAD_DIM_V]` accumulator
/// in `sdpa.cu`. `head_dim_v` indexes it directly with no bounds check, so a
/// larger value corrupts the stack — the launcher refuses it.
pub(super) const SDPA_MAX_HEAD_DIM_V: usize = 256;

/// Which compiled tile a `(head_dim_k, head_dim_v)` request resolves to.
///
/// The variant, not a bare `bool`, is what determines the kernel-name suffix.
/// This mirrors `varlen_attention_block_config::TileVariant`, which exists
/// because a bool-based selector once produced a kernel name that was never
/// compiled (`CUDA_ERROR_NOT_FOUND`): the suffix is derived from the variant
/// in exactly one place ([`TileVariant::suffix`]) and never re-decided at the
/// call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileVariant {
    /// Unsuffixed large tile (`BLOCK_M=128, BLOCK_N=128`).
    Large,
    /// `_small`-suffixed fallback tile (`BLOCK_M=64, BLOCK_N=32`).
    Small,
}

impl TileVariant {
    /// `(BLOCK_M, BLOCK_N)` of this variant. Must stay in sync with the
    /// `SDPA_ENTRY` instantiations in `sdpa.cu`.
    fn dims(self) -> (usize, usize) {
        match self {
            TileVariant::Large => (128, 128),
            TileVariant::Small => (64, 32),
        }
    }

    /// Kernel-name suffix for this variant — the ONLY place that maps a
    /// variant to a suffix, so a resolved tile and the symbol loaded for it
    /// cannot drift apart.
    fn suffix(self) -> &'static str {
        match self {
            TileVariant::Large => "",
            TileVariant::Small => "_small",
        }
    }
}

/// A resolved SDPA launch configuration.
///
/// The kernel name, the block dimension and the shared-memory request are all
/// produced together by [`block_config`] from one resolved [`TileVariant`], so
/// the launcher cannot pair a name with a tile it was not built for. `block_m`
/// is both the Q rows per block and the block's thread count — `sdpa.cu` gives
/// one Q row to each thread (`is_valid_thread = tid < q_tile_size`).
pub(super) struct SdpaTile {
    /// Symbol to load out of `SDPA_MODULE`.
    pub(super) kernel_name: String,
    /// Q rows per block, and `block_dim.x`.
    pub(super) block_m: usize,
    /// K/V columns staged per iteration.
    pub(super) block_n: usize,
    /// Dynamic shared memory to opt in to and to request at launch.
    pub(super) smem_bytes: usize,
}

/// Base kernel name for a dtype, without the tile suffix.
fn dtype_base_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("sdpa_f32"),
        DType::F16 => Ok("sdpa_f16"),
        DType::BF16 => Ok("sdpa_bf16"),
        _ => Err(Error::KernelError {
            reason: format!("SDPA: unsupported dtype {:?}", dtype),
        }),
    }
}

/// Shared memory bytes the SDPA kernel needs for a given tile and shape.
///
/// Layout in `sdpa.cu` (`sdpa_impl`):
/// `[Q: BLOCK_M x head_dim_k][K: BLOCK_N x head_dim_k][V: BLOCK_N x head_dim_v]`,
/// with NO `+1` bank-conflict padding — unlike the flash forward layout in
/// `flash_utils::compute_smem`, which must NOT be reused here.
///
/// Every instantiation declares the tiles as `float*` and converts F16/BF16
/// inputs to `float` on load, so the element size is always `f32` and NOT the
/// input dtype. The requirement is a function of the SHAPE and the TILE alone.
pub(super) fn sdpa_smem_size(
    block_m: usize,
    block_n: usize,
    head_dim_k: usize,
    head_dim_v: usize,
) -> usize {
    (block_m * head_dim_k + block_n * head_dim_k + block_n * head_dim_v) * size_of::<f32>()
}

/// Pick the SDPA tile that fits this device's opt-in shared memory.
///
/// Tries the large `(128, 128)` tile first and falls back to the small
/// `(64, 32)` tile when the large one exceeds `device_max_smem()`. At a
/// DeepSeek-V2/V3-shaped MLA (`head_dim_k = head_dim + rope_head_dim = 192`,
/// `head_dim_v = 128`) that is 262144 bytes large versus 90112 bytes small,
/// so the small tile is what makes the shape runnable on a device whose
/// opt-in limit sits below 256KB.
///
/// No device figure is baked in: the limit is queried per call.
pub(super) fn block_config(dtype: DType, head_dim_k: usize, head_dim_v: usize) -> Result<SdpaTile> {
    let base = dtype_base_name(dtype)?;
    let max_smem = device_max_smem();

    for variant in [TileVariant::Large, TileVariant::Small] {
        let (block_m, block_n) = variant.dims();
        let smem_bytes = sdpa_smem_size(block_m, block_n, head_dim_k, head_dim_v);
        if smem_bytes <= max_smem {
            return Ok(SdpaTile {
                kernel_name: format!("{}{}", base, variant.suffix()),
                block_m,
                block_n,
                smem_bytes,
            });
        }
    }

    let (small_m, small_n) = TileVariant::Small.dims();
    let smallest = sdpa_smem_size(small_m, small_n, head_dim_k, head_dim_v);
    Err(Error::KernelError {
        reason: format!(
            "SDPA shared memory requirement ({} bytes) exceeds the device opt-in limit \
             ({} bytes) for head_dim_k={}, head_dim_v={}: that is the smallest tile \
             sdpa.cu instantiates (BLOCK_M={}, BLOCK_N={}), so there is no smaller tile \
             to fall back to",
            smallest, max_smem, head_dim_k, head_dim_v, small_m, small_n
        ),
    })
}

/// Test-only entry point exposing [`block_config`]'s decision together with
/// this device's opt-in limit, so tile-selection tests can assert on the
/// RESOLVED tile rather than on a launch merely succeeding — without reaching
/// into the crate-private helpers. Mirrors
/// `paged_attention_fwd_block_config::fwd_prefill_tile_for_test`.
///
/// Returns `(kernel_name, block_m, block_n, smem_bytes, device_max_smem_bytes)`.
#[doc(hidden)]
pub fn mla_tile_for_test(
    dtype: DType,
    head_dim_k: usize,
    head_dim_v: usize,
) -> Result<(String, usize, usize, usize, usize)> {
    let tile = block_config(dtype, head_dim_k, head_dim_v)?;
    Ok((
        tile.kernel_name,
        tile.block_m,
        tile.block_n,
        tile.smem_bytes,
        device_max_smem(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smem_is_dtype_independent_and_matches_the_documented_arithmetic() {
        // DeepSeek-V2/V3-shaped MLA: head_dim_k=192, head_dim_v=128.
        assert_eq!(sdpa_smem_size(128, 128, 192, 128), 262144);
        assert_eq!(sdpa_smem_size(64, 32, 192, 128), 90112);
        // Square 128/128.
        assert_eq!(sdpa_smem_size(128, 128, 128, 128), 196608);
        assert_eq!(sdpa_smem_size(64, 32, 128, 128), 65536);
    }

    #[test]
    fn suffix_is_the_single_mapping_point() {
        assert_eq!(TileVariant::Large.suffix(), "");
        assert_eq!(TileVariant::Small.suffix(), "_small");
        assert_eq!(TileVariant::Large.dims(), (128, 128));
        assert_eq!(TileVariant::Small.dims(), (64, 32));
    }
}
