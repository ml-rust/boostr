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
    let device = act_contig.device();
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
    use super::super::formats::Q8_0;
    use super::*;

    /// Ample limit: every compiled variant fits.
    const WIDE: u32 = 1 << 20;

    #[test]
    fn selects_the_smallest_tile_that_covers_one_batch() {
        assert_eq!(select_variant(1, WIDE, &Q8_0), Some(8));
        assert_eq!(select_variant(8, WIDE, &Q8_0), Some(8));
        assert_eq!(select_variant(9, WIDE, &Q8_0), Some(16));
        assert_eq!(select_variant(128, WIDE, &Q8_0), Some(128));
    }

    #[test]
    fn ties_go_to_the_smaller_tile() {
        // 129 needs two tiles at every variant from 80 up, so the scan keeps 80.
        assert_eq!(select_variant(129, WIDE, &Q8_0), Some(80));
    }

    #[test]
    fn honours_the_shared_memory_limit() {
        // Only the smallest variants fit under a limit set just above x8.
        assert_eq!(select_variant(1024, smem_bytes(&Q8_0, 8), &Q8_0), Some(8));
        assert_eq!(select_variant(1024, smem_bytes(&Q8_0, 24), &Q8_0), Some(24));
        assert_eq!(select_variant(1024, 0, &Q8_0), None);
    }

    #[test]
    fn shared_memory_grows_only_with_the_token_tile() {
        assert_eq!(smem_bytes(&Q8_0, 8), 4 * (128 * 76 + 8 * 36));
        assert_eq!(smem_bytes(&Q8_0, 128), 57344);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q8_0, x) <= smem_bytes(&Q8_0, 128))
        );
    }
}
