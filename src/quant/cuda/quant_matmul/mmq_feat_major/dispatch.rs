use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaFunction, CudaModule, LaunchConfig};
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use super::super::super::kernels::{self, QUANT_MMQ_MMA_MODULE};
use super::super::helpers::quantize_activation_q8_1_mmq;
use super::formats::FeatMajorFormat;

/// The activation contract every kernel in this family satisfies.
///
/// [`dispatch`] repacks the caller's activation into the 8-bit dynamic record
/// `quantize_activation_q8_1_mmq` builds — 8-bit codes per group of 32 values
/// along K — and the MMA instructions accumulate on integers before rescaling
/// to f32. The activation the dot product sees is NOT the activation the
/// caller handed in.
///
/// Reassociates: `mma.sync` reduces inside the instruction, and the split-K
/// fixup pass above (`use_stream_k`) reduces again across tiles.
pub(in crate::quant::cuda::quant_matmul) const CONTRACT: crate::quant::KernelContract =
    crate::quant::KernelContract::dynamic_int8_activation("cuda feature-major MMQ");

/// Output features per block, fixed by the kernel's weight tile.
pub(super) const FEAT_TILE: u32 = 128;

/// Threads per block, fixed by the kernel.
const THREADS: u32 = 256;

/// Compiled token-tile variants, ascending. Below 48 the tile steps by 8, at and
/// above it by 16; the kernel's warp blocking rejects every other value.
pub(super) const VARIANTS: &[u32] = &[8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Activation row stride in the shared tile, in ints: 4 half2 scale pairs plus
/// 32 quant words. The same for every weight format.
const ACT_STRIDE: u32 = 36;

/// Dynamic shared memory one variant needs: a `FEAT_TILE`-row weight tile at
/// the format's stride, an `mmq_x`-row activation tile at `ACT_STRIDE`, and the
/// format's per-token activation scratch, which is zero for every format whose
/// minimum term is no finer than the record's 32-value sub-block.
///
/// Takes the whole descriptor rather than a stride so a format cannot be
/// launched with less shared memory than its kernel indexes.
pub(super) const fn smem_bytes(format: &FeatMajorFormat, mmq_x: u32) -> u32 {
    4 * (FEAT_TILE * format.x_stride + mmq_x * (ACT_STRIDE + format.act_scratch_ints_per_token))
}

/// Per-block dynamic shared-memory ceiling this device grants on opt-in.
///
/// The per-block attribute reports only the static default, which every variant
/// above the smallest exceeds. The per-SM figure less the driver's reservation
/// is the bound that actually applies once a function opts in.
pub(super) fn smem_opt_in_limit(shared_mem_per_unit: u32) -> u32 {
    shared_mem_per_unit.saturating_sub(1024)
}

/// Picks the token tile that launches the fewest tiles for `m`, breaking ties
/// toward the smaller tile because it costs fewer registers and less shared
/// memory. `None` means no variant fits the device.
fn select_variant(m: u32, smem_limit: u32, format: &FeatMajorFormat) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in VARIANTS {
        if smem_bytes(format, mmq_x) > smem_limit {
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

/// Whether some compiled variant serves `m` tokens of `format` on this
/// device — the same test [`dispatch`] applies before it quantizes.
pub(in crate::quant::cuda::quant_matmul) fn variant_fits(
    format: &FeatMajorFormat,
    m: usize,
    device_index: usize,
) -> bool {
    let profile = CudaDevice::new(device_index).profile();
    select_variant(
        m as u32,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        format,
    )
    .is_some()
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

/// The activation record every variant can read: token slots padded to the
/// widest token tile.
///
/// A variant's staging copies `mmq_x` records per tile from `tok0`, so a
/// buffer padded to a multiple of any larger tile holds every tile's copy.
/// One activation quantized this way serves several weights whose variants
/// differ, which is what `quant_matmul_batch` needs: the record is 9/8 of
/// the f32 row, so the padding costs at most one widest tile of tokens.
///
/// Returns the buffer and its token stride, the `ntok` the kernels take.
pub(in crate::quant::cuda::quant_matmul) fn quantize_shared_activation(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    m: usize,
    k: usize,
) -> Result<(Tensor<CudaRuntime>, u32)> {
    let widest = *VARIANTS.last().unwrap_or(&FEAT_TILE) as usize;
    quantize_activation_q8_1_mmq(client, act_contig, m, k, widest)
}

/// Runs one quantized weight x F32 activation through the feature-major MMQ
/// kernels. `Ok(None)` means no variant fits, and the caller should keep its
/// existing path.
#[allow(clippy::too_many_arguments)]
pub(in crate::quant::cuda::quant_matmul) fn dispatch(
    format: &FeatMajorFormat,
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();
    let profile = CudaDevice::new(device_index).profile();
    let Some(mmq_x) = select_variant(
        m as u32,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        format,
    ) else {
        return Ok(None);
    };

    // This path reads its own activation layout, k-group-major and
    // token-minor, so the tile copy is flat. The per-token producer stays
    // untouched for `quant_mmq_q8_0_q8_1_mma`, dp4a and the K-quants.
    let (q8_buf, ntok) = quantize_activation_q8_1_mmq(client, act_contig, m, k, mmq_x as usize)?;
    dispatch_quantized(
        format,
        client,
        act_contig.device(),
        q8_buf.ptr(),
        ntok,
        weight,
        output_ptr,
        m,
        k,
        n,
    )
}

/// [`dispatch`] with the activation already in the MMQ record layout at
/// `q8_ptr` with token stride `ntok` — from [`quantize_shared_activation`],
/// or from a producer sized for this call's own variant.
#[allow(clippy::too_many_arguments)]
pub(in crate::quant::cuda::quant_matmul) fn dispatch_quantized(
    format: &FeatMajorFormat,
    client: &CudaClient,
    device: &CudaDevice,
    q8_ptr: u64,
    ntok: u32,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device_index = device.id();
    let profile = CudaDevice::new(device_index).profile();

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let Some(mmq_x) = select_variant(
        m_u32,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        format,
    ) else {
        return Ok(None);
    };
    let smem = smem_bytes(format, mmq_x);

    let token_tiles = m_u32.div_ceil(mmq_x);
    let feat_tiles = n_u32.div_ceil(FEAT_TILE);
    let tiles = token_tiles * feat_tiles;
    let sms = profile.compute_units;

    // The record's token stride must cover this variant's last tile copy.
    if ntok < token_tiles * mmq_x {
        return Err(Error::QuantError {
            reason: format!(
                "MMQ activation record has {ntok} token slots, variant x{mmq_x} over {m} tokens needs {}",
                token_tiles * mmq_x
            ),
        });
    }
    let weight_ptr = weight.storage().ptr();

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)?;

    let stream_k = use_stream_k(tiles, sms, k_u32, format);

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

/// Activation k-blocks per token, the kernel's own work-unit divisor.
///
/// `mmqf_sk_body` in `src/quant/cuda/kernels/quant_mmq_mma.cu` slices
/// `total = ntf * ntt * nbk` across `gridDim.x`, where `nbk` is `K / 32`.
/// The grid clamp here mirrors that count so no block is launched with no
/// work, which means the two must agree on this divisor.
const SK_K_STEP: u32 = 32;

/// Launches the stream-k pair. Both grids are the driver's occupancy-derived
/// block count: the fixup rebuilds every other block's slice bounds from its
/// own index and `gridDim.x`, so the two launches must agree on the grid.
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
    let sk_name = format!("quant_mmq_{}_q8_1_mma_sk_x{mmq_x}", format.kernel_infix);
    let sk_func = kernels::get_kernel_function(module, &sk_name)?;
    opt_in_shared(&sk_func, smem, &sk_name)?;

    // The opt-in must run first: occupancy at the default shared-memory
    // limit undercounts blocks for every variant above the smallest.
    let blocks_per_sm = sk_func
        .occupancy_max_active_blocks_per_multiprocessor(THREADS, smem as usize, None)
        .unwrap_or(1)
        .max(1);

    let k_blocks = k_u32.div_ceil(SK_K_STEP);
    let blocks = (sms * blocks_per_sm).min(tiles * k_blocks).max(1);

    // A block's slice starts on a tile boundary exactly when the tile count
    // divides the grid, and then no block is left holding a partial tile.
    let fixup_needed = !tiles.is_multiple_of(blocks);

    // Never zeroed: the fixup reads only slots whose block provably wrote a
    // partial, so a memset would be pure cost. When no partial can exist the
    // buffer is a placeholder that keeps the argument a valid allocation.
    let ws_len = if fixup_needed {
        blocks as usize * mmq_x as usize * FEAT_TILE as usize
    } else {
        1
    };
    let ws = Tensor::<CudaRuntime>::empty(&[ws_len], DType::F32, device)?;
    let ws_ptr = ws.ptr();

    let cfg_sk = LaunchConfig {
        grid_dim: (blocks, 1, 1),
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
        grid_dim: (blocks, 1, 1),
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

/// Whether to launch the stream-k pair rather than the tile-parallel grid.
///
/// Stream-k splits every tile's K dimension across blocks and pays a fixup
/// pass to rejoin the partials, trading that pass for the wave a ragged tile
/// count leaves half empty. Once the tiles fill the device, tile-parallel
/// wins and needs no workspace.
///
/// A format vetoes this call through `prefers_tile_parallel`, but only once
/// the tile count passes about four thirds of the SM count: past that point
/// the split saves too little to cover the fixup pass. Below that threshold
/// the tile-parallel grid cannot fill the device, and stream-k wins for every
/// format, veto or not.
///
/// K gates it too. The partial stores and the fixup pass are a fixed cost per
/// split, paid once however short the K walk is, so a short K cannot amortise
/// them and the tile-parallel grid wins even with most SMs idle. Measured on
/// every K-quant and IQ4 format at the DiT projection shapes: below
/// `STREAM_K_MIN_K` stream-k loses for every format at every tile count
/// tried; at and above it stream-k wins.
const fn use_stream_k(tiles: u32, sms: u32, k: u32, format: &FeatMajorFormat) -> bool {
    sms > 0
        && k >= STREAM_K_MIN_K
        && tiles < 2 * sms
        && !(format.prefers_tile_parallel && 3 * tiles >= 4 * sms)
}

/// Shortest K the stream-k split is worth. See [`use_stream_k`].
const STREAM_K_MIN_K: u32 = 2048;

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
