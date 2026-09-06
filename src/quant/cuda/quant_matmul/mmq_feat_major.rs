//! Tensor-core MMQ dispatch, feature-major tiling.
//!
//! The output-feature dimension gets the fixed 128 tile and the weight is the
//! MMA operand A. `quant_mmq_q8_0_q8_1_mma` fixes the token tile instead and
//! makes the activation operand A; this path swaps those roles. One entry
//! point is compiled per (weight format, token tile), in three roles:
//! tile-parallel, stream-k, and the stream-k fixup. This module owns the rules
//! that choose among them. The kernels themselves live in
//! `src/quant/cuda/kernels/quant_mmq_mma.cu`.
//!
//! The kernel family is parameterized over the weight format; everything that
//! differs per format is a field of `FeatMajorFormat`. Q8_0, Q4_0, Q4_K,
//! Q5_K and Q6_K are the formats compiled today.
//!
//! This path needs sm_80 and its own repacked activation layout, so the caller
//! gates on `caps.int8_mma_m16n8k32` and falls back to `quant_mmq_q8_0_q8_1_mma`
//! when this returns `Ok(None)`.

use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaFunction, CudaModule, LaunchConfig};
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{self, QUANT_MMQ_MMA_MODULE};
use super::helpers::quantize_activation_q8_1_mmq;

/// Output features per block, fixed by the kernel's weight tile.
const FEAT_TILE: u32 = 128;

/// Threads per block, fixed by the kernel.
const THREADS: u32 = 256;

/// Compiled token-tile variants, ascending. Below 48 the tile steps by 8, at and
/// above it by 16; the kernel's warp blocking rejects every other value.
const VARIANTS: &[u32] = &[8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Activation row stride in the shared tile, in ints: 4 half2 scale pairs plus
/// 32 quant words. The same for every weight format.
const ACT_STRIDE: u32 = 36;

/// One weight format's share of the feature-major family. Everything else in
/// this module — variant choice, stream-k decision, launch, fixup — is shared.
pub(super) struct FeatMajorFormat {
    /// Format name inside the kernel symbol, as `MMQ_FM_KERNEL`'s `NAME`.
    pub kernel_infix: &'static str,
    /// Weight row stride in the shared tile, in ints (`FMT::X_STRIDE`).
    pub x_stride: u32,
    /// K must be a whole number of these for the staging map to hold.
    pub k_multiple: u32,
}

/// Q8_0: 34-byte blocks of 32 elements, staged as 64 quant words plus 8 f32
/// scales plus 4 ints of bank padding.
pub(super) const Q8_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q8_0",
    x_stride: 76,
    k_multiple: 32,
};

/// Q4_0: 18-byte blocks of 32 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 scales plus 4 ints of bank padding. Q4_0's quants
/// are unsigned 4-bit biased by 8, and the kernel folds that bias in while
/// staging, so the staged row and the whole `vec_dot` are Q8_0's. K needs only
/// a whole 32-element block, so a row's last 256-k staging group can be
/// partial.
pub(super) const Q4_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q4_0",
    x_stride: 76,
    k_multiple: 32,
};

/// Q4_K: 144-byte super-blocks of 256 elements, staged as 64 quant words plus
/// 8 `float2` scale/min pairs (16 ints) plus 4 ints of bank padding. The row is
/// 8 ints wider than Q8_0's because the pair is f32, not `half2`: half rounding
/// on `d * sc` perturbs every 32-element sub-block and pushed the GEMM path
/// outside the GEMV parity bound. K must be a whole number of super-blocks,
/// which also makes every 256-k staging group whole.
pub(super) const Q4_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q4_k",
    x_stride: 84,
    k_multiple: 256,
};

/// Q5_K: 176-byte super-blocks of 256 elements, staged exactly as Q4_K — 64
/// quant words plus 8 `float2` scale/min pairs (16 ints) plus 4 ints of bank
/// padding. Q5_K is Q4_K with a fifth quant bit from a 32-byte `qh` field, so
/// only the kernel's staging step differs; the staged row and the two-term
/// arithmetic are shared. K must be a whole number of super-blocks.
pub(super) const Q5_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q5_k",
    x_stride: 84,
    k_multiple: 256,
};

/// Q6_K: 210-byte super-blocks of 256 elements, staged as 64 quant words plus
/// 16 f32 group scales plus 4 ints of bank padding — the same stride as Q4_K,
/// reached by a different split. Q6_K's scale changes every 16 elements, so it
/// stages 16 `d * scale` floats per row rather than 8 scale/min pairs, and the
/// consumer runs two 16-k MMAs per 32-k step. The staged scale is f32 and
/// already multiplied by `d`, for the same parity reason as Q4_K. K must be a
/// whole number of super-blocks.
pub(super) const Q6_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q6_k",
    x_stride: 84,
    k_multiple: 256,
};

/// Dynamic shared memory one variant needs: a `FEAT_TILE`-row weight tile at
/// the format's stride plus an `mmq_x`-row activation tile at `ACT_STRIDE`.
const fn smem_bytes(x_stride: u32, mmq_x: u32) -> u32 {
    4 * (FEAT_TILE * x_stride + mmq_x * ACT_STRIDE)
}

/// Per-block dynamic shared-memory ceiling this device grants on opt-in.
///
/// The per-block attribute reports only the static default, which every variant
/// above the smallest exceeds. The per-SM figure less the driver's reservation
/// is the bound that actually applies once a function opts in.
fn smem_opt_in_limit(shared_mem_per_unit: u32) -> u32 {
    shared_mem_per_unit.saturating_sub(1024)
}

/// Picks the token tile that launches the fewest tiles for `m`, breaking ties
/// toward the smaller tile because it costs fewer registers and less shared
/// memory. `None` means no variant fits the device.
fn select_variant(m: u32, smem_limit: u32, x_stride: u32) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in VARIANTS {
        if smem_bytes(x_stride, mmq_x) > smem_limit {
            continue;
        }
        let tiles = m.div_ceil(mmq_x);
        if best.is_none_or(|(_, best_tiles)| tiles < best_tiles) {
            best = Some((mmq_x, tiles));
        }
        if tiles == 1 {
            break;
        }
    }
    best.map(|(mmq_x, _)| mmq_x)
}

/// Opts a function in to more than the static shared-memory limit. Required
/// before the first launch of every variant; the limit is per function.
fn opt_in_shared(func: &CudaFunction, bytes: u32, name: &str) -> Result<()> {
    func.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        bytes as i32,
    )
    .map_err(|e| Error::QuantError {
        reason: format!("CUDA {name} shared-memory opt-in failed: {e:?}"),
    })
}

/// Runs one quantized weight x F32 activation through the feature-major MMQ
/// kernels. `Ok(None)` means no variant fits, and the caller should keep its
/// existing path.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch(
    format: &FeatMajorFormat,
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device = act_contig.device();
    let device_index = device.id();
    let profile = CudaDevice::new(device_index).profile();

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let Some(mmq_x) = select_variant(
        m_u32,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        format.x_stride,
    ) else {
        return Ok(None);
    };
    let smem = smem_bytes(format.x_stride, mmq_x);

    let token_tiles = m_u32.div_ceil(mmq_x);
    let feat_tiles = n_u32.div_ceil(FEAT_TILE);
    let tiles = token_tiles * feat_tiles;
    let sms = profile.compute_units;

    // This path reads its own activation layout, k-group-major and
    // token-minor, so the tile copy is flat. The per-token producer stays
    // untouched for `quant_mmq_q8_0_q8_1_mma`, dp4a and the K-quants.
    let (q8_buf, ntok) = quantize_activation_q8_1_mmq(client, act_contig, m, k, mmq_x as usize)?;
    let q8_ptr = q8_buf.ptr();
    let weight_ptr = weight.storage().ptr();

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)?;

    // Stream-k when the tile count alone leaves the device short of work: it
    // launches one block per SM and splits K across them. Once the tiles fill
    // the device the tile-parallel grid is the better shape and needs no
    // workspace or fixup pass.
    let stream_k = sms > 0 && tiles < 2 * sms;

    tracing::debug!(
        m,
        k,
        n,
        mmq_x,
        tiles,
        stream_k,
        weight_format = format.kernel_infix,
        path = "mmq_feat_major",
        "CUDA quant kernel: tensor-core MMQ (feature-major)"
    );

    if stream_k {
        launch_stream_k(
            format, client, device, &module, output_ptr, q8_ptr, weight_ptr, m_u32, k_u32, n_u32,
            ntok, mmq_x, smem, sms, tiles,
        )?;
    } else {
        let name = format!("quant_mmq_{}_q8_1_mma_x{mmq_x}", format.kernel_infix);
        let func = kernels::get_kernel_function(&module, &name)?;
        opt_in_shared(&func, smem, &name)?;

        let cfg = LaunchConfig {
            grid_dim: (token_tiles, feat_tiles, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&q8_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&output_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.arg(&ntok);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA {name} launch failed: {e:?}"),
            })?;
        }
    }

    Ok(Some(()))
}

/// Launches the stream-k pair. Both grids are the SM count: the fixup rebuilds
/// every other block's slice bounds from its own index and `gridDim.x`, so the
/// two launches must agree on the grid.
#[allow(clippy::too_many_arguments)]
fn launch_stream_k(
    format: &FeatMajorFormat,
    client: &CudaClient,
    device: &CudaDevice,
    module: &std::sync::Arc<CudaModule>,
    output_ptr: u64,
    q8_ptr: u64,
    weight_ptr: u64,
    m_u32: u32,
    k_u32: u32,
    n_u32: u32,
    ntok: u32,
    mmq_x: u32,
    smem: u32,
    sms: u32,
    tiles: u32,
) -> Result<()> {
    // A block's slice starts on a tile boundary exactly when the tile count
    // divides the grid, and then no block is left holding a partial tile.
    let fixup_needed = !tiles.is_multiple_of(sms);

    // Never zeroed: the fixup reads only slots whose block provably wrote a
    // partial, so a memset would be pure cost. When no partial can exist the
    // buffer is a placeholder that keeps the argument a valid allocation.
    let ws_len = if fixup_needed {
        sms as usize * mmq_x as usize * FEAT_TILE as usize
    } else {
        1
    };
    let ws = Tensor::<CudaRuntime>::empty(&[ws_len], DType::F32, device)?;
    let ws_ptr = ws.ptr();

    let sk_name = format!("quant_mmq_{}_q8_1_mma_sk_x{mmq_x}", format.kernel_infix);
    let sk_func = kernels::get_kernel_function(module, &sk_name)?;
    opt_in_shared(&sk_func, smem, &sk_name)?;

    let cfg_sk = LaunchConfig {
        grid_dim: (sms, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&sk_func);
        builder.arg(&q8_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&ws_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.arg(&ntok);
        builder.launch(cfg_sk).map_err(|e| Error::QuantError {
            reason: format!("CUDA {sk_name} launch failed: {e:?}"),
        })?;
    }

    if !fixup_needed {
        return Ok(());
    }

    // A separate launch on the same stream: the fixup reads what the main
    // kernel wrote, so the two must not be fused.
    let fx_name = format!("quant_mmq_{}_q8_1_mma_fixup_x{mmq_x}", format.kernel_infix);
    let fx_func = kernels::get_kernel_function(module, &fx_name)?;
    let cfg_fx = LaunchConfig {
        grid_dim: (sms, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&fx_func);
        builder.arg(&output_ptr);
        builder.arg(&ws_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg_fx).map_err(|e| Error::QuantError {
            reason: format!("CUDA {fx_name} launch failed: {e:?}"),
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ample limit: every compiled variant fits.
    const WIDE: u32 = 1 << 20;

    /// The stride Q8_0 and Q4_0 share.
    const XS: u32 = Q8_0.x_stride;

    #[test]
    fn selects_the_smallest_tile_that_covers_one_batch() {
        assert_eq!(select_variant(1, WIDE, XS), Some(8));
        assert_eq!(select_variant(8, WIDE, XS), Some(8));
        assert_eq!(select_variant(9, WIDE, XS), Some(16));
        assert_eq!(select_variant(128, WIDE, XS), Some(128));
    }

    #[test]
    fn ties_go_to_the_smaller_tile() {
        // 129 needs two tiles at every variant from 80 up, so the scan keeps 80.
        assert_eq!(select_variant(129, WIDE, XS), Some(80));
    }

    #[test]
    fn honours_the_shared_memory_limit() {
        // Only the smallest variants fit under a limit set just above x8.
        assert_eq!(select_variant(1024, smem_bytes(XS, 8), XS), Some(8));
        assert_eq!(select_variant(1024, smem_bytes(XS, 24), XS), Some(24));
        assert_eq!(select_variant(1024, 0, XS), None);
    }

    #[test]
    fn shared_memory_grows_only_with_the_token_tile() {
        assert_eq!(smem_bytes(XS, 8), 4 * (128 * 76 + 8 * 36));
        assert_eq!(smem_bytes(XS, 128), 57344);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(XS, x) <= smem_bytes(XS, 128))
        );
    }

    #[test]
    fn the_q8_0_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q8_0.kernel_infix, 8),
            "quant_mmq_q8_0_q8_1_mma_x8"
        );
        assert_eq!(Q8_0.k_multiple, 32);
    }

    /// Q4_0 stages into the Q8_0 row, so the two strides must stay equal and
    /// with them the family's shared-memory request at every token tile. Both
    /// are 32-element block formats, so both take the ragged-K multiple.
    #[test]
    fn the_q4_0_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q4_0.kernel_infix, 8),
            "quant_mmq_q4_0_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q4_0.kernel_infix, 128),
            "quant_mmq_q4_0_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q4_0.kernel_infix, 128),
            "quant_mmq_q4_0_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q4_0.k_multiple, 32);
        assert_eq!(Q4_0.x_stride, Q8_0.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(Q4_0.x_stride, x) == smem_bytes(Q8_0.x_stride, x))
        );
    }

    #[test]
    fn the_q4_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q4_K.kernel_infix, 8),
            "quant_mmq_q4_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q4_K.kernel_infix, 128),
            "quant_mmq_q4_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q4_K.kernel_infix, 128),
            "quant_mmq_q4_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q4_K.k_multiple, 256);
    }

    #[test]
    fn the_q5_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q5_K.kernel_infix, 8),
            "quant_mmq_q5_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q5_K.kernel_infix, 128),
            "quant_mmq_q5_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q5_K.kernel_infix, 128),
            "quant_mmq_q5_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q5_K.k_multiple, 256);
    }

    /// Q5_K stages the Q4_K row verbatim — same quant words, same eight
    /// scale/min pairs — so the two strides must stay equal, and with them the
    /// family's shared-memory request at every token tile.
    #[test]
    fn q5_k_shares_the_q4_k_row_stride() {
        assert_eq!(Q5_K.x_stride, Q4_K.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(Q5_K.x_stride, x) == smem_bytes(Q4_K.x_stride, x))
        );
    }

    #[test]
    fn the_q6_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q6_K.kernel_infix, 8),
            "quant_mmq_q6_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q6_K.kernel_infix, 128),
            "quant_mmq_q6_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q6_K.kernel_infix, 128),
            "quant_mmq_q6_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q6_K.k_multiple, 256);
    }

    /// Q6_K reaches Q4_K's row stride by a different split: 16 f32 group
    /// scales rather than 8 scale/min pairs. Equal strides mean the two share
    /// the family's shared-memory request at every token tile.
    #[test]
    fn q6_k_shares_the_q4_k_row_stride() {
        assert_eq!(Q6_K.x_stride, Q4_K.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(Q6_K.x_stride, x) == smem_bytes(Q4_K.x_stride, x))
        );
    }

    /// Q4_K's weight row is 8 ints wider than Q8_0's: it stages the scale/min
    /// pair as two f32 rather than one `half2`, because half rounding on
    /// `d * sc` broke the GEMM/GEMV parity bound. The 8 extra ints per row
    /// across the 128-row weight tile are the whole difference in the request,
    /// and it is the same at every token tile because only the activation tile
    /// scales with `mmq_x`.
    #[test]
    fn q4_k_costs_one_extra_scale_word_per_row_over_q8_0() {
        const EXTRA: u32 = 4 * FEAT_TILE * 8;
        assert_eq!(Q4_K.x_stride, Q8_0.x_stride + 8);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| { smem_bytes(Q4_K.x_stride, x) == smem_bytes(Q8_0.x_stride, x) + EXTRA })
        );
        // The widest variant must still fit what a device grants on opt-in.
        // 96KB per unit is the smallest sm_80-or-later figure the family runs
        // on, and the launcher subtracts the driver's reservation from it.
        assert!(smem_bytes(Q4_K.x_stride, 128) <= smem_opt_in_limit(96 * 1024));
    }
}
