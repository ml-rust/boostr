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
use super::tiling::{
    Cadence, FEAT_TILE_DEFAULT, FeatTile, Role, Tiling, record_token_slots, select_tiling,
    select_variant, smem_opt_in_limit, split_count, use_split_launch,
};

/// Whether some compiled variant serves `m` tokens of `format` on this
/// device — the same test [`dispatch`] applies before it quantizes.
///
/// Asks for the default feature tile: the automatic rule only reaches the
/// narrow tile through a default tiling that fits, so this is the gate for
/// both.
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
        FEAT_TILE_DEFAULT,
        Cadence::Halves,
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

/// The activation record every variant of every format in `formats` can
/// read for `m` tokens: token slots padded to the most any of their compiled
/// tilings' token tiles cover.
///
/// A variant's staging copies `mmq_x` records per tile from `tok0`, so a
/// buffer padded to that many slots holds every tile's copy. One activation
/// quantized this way serves several weights whose variants differ, which is
/// what `quant_matmul_batch` needs.
///
/// Returns the buffer and its token stride, the `ntok` the kernels take.
pub(in crate::quant::cuda::quant_matmul) fn quantize_shared_activation(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    formats: &[&FeatMajorFormat],
    m: usize,
    k: usize,
) -> Result<(Tensor<CudaRuntime>, u32)> {
    let profile = CudaDevice::new(act_contig.device().id()).profile();
    let limit = smem_opt_in_limit(profile.shared_mem_per_unit);
    let slots = formats
        .iter()
        .map(|format| record_token_slots(m as u32, limit, format))
        .max()
        .unwrap_or(0)
        .max(m as u32);
    // `quantize_activation_q8_1_mmq` pads to a multiple of its tile argument;
    // the slot count is already that multiple of every tile it covers.
    quantize_activation_q8_1_mmq(client, act_contig, m, k, slots as usize)
}

/// The tiling [`dispatch`] and [`dispatch_quantized`] agree on for one call.
/// Both compute it from the same inputs, so the record the first quantizes
/// covers the tile the second launches.
fn tiling_for(
    format: &FeatMajorFormat,
    profile_index: usize,
    m: usize,
    n: usize,
    k: usize,
    feat_tile: FeatTile,
) -> Result<Option<Tiling>> {
    let profile = CudaDevice::new(profile_index).profile();
    select_tiling(
        m as u32,
        n as u32,
        k as u32,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        profile.compute_units,
        format,
        feat_tile,
    )
}

/// Runs one quantized weight x F32 activation through the feature-major MMQ
/// kernels. `Ok(None)` means no variant fits, and the caller should keep its
/// existing path.
///
/// `feat_tile` is [`FeatTile::Auto`] for every production caller; a forced
/// tile is the kernel A/B's hook and errors when the format lacks that tile.
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
    feat_tile: FeatTile,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();
    let Some(tiling) = tiling_for(format, device_index, m, n, k, feat_tile)? else {
        return Ok(None);
    };

    // This path reads its own activation layout, k-group-major and
    // token-minor, so the tile copy is flat. The per-token producer stays
    // untouched for `quant_mmq_q8_0_q8_1_mma`, dp4a and the K-quants.
    let (q8_buf, ntok) =
        quantize_activation_q8_1_mmq(client, act_contig, m, k, tiling.mmq_x as usize)?;
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
        feat_tile,
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
    feat_tile: FeatTile,
) -> Result<Option<()>> {
    let device_index = device.id();
    let profile = CudaDevice::new(device_index).profile();

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let Some(tiling) = tiling_for(format, device_index, m, n, k, feat_tile)? else {
        return Ok(None);
    };
    let smem = tiling.smem_bytes(format);

    let token_tiles = tiling.token_tiles(m_u32);
    let feat_tiles = tiling.feat_tiles(n_u32);
    let tiles = token_tiles * feat_tiles;
    let sms = profile.compute_units;

    // The record's token stride must cover this variant's last tile copy.
    if ntok < token_tiles * tiling.mmq_x {
        return Err(Error::QuantError {
            reason: format!(
                "MMQ activation record has {ntok} token slots, variant x{} over {m} tokens needs {}",
                tiling.mmq_x,
                token_tiles * tiling.mmq_x
            ),
        });
    }
    let weight_ptr = weight.storage().ptr();

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)?;

    // The split count reads K, N and the device only, so the float sequence
    // each output element receives is fixed before M or the tiling is known.
    let splits = split_count(k_u32, n_u32, sms);
    let split_launch = use_split_launch(splits, tiles, tiling.feat_tile, sms, format);

    tracing::debug!(
        m,
        k,
        n,
        feat_tile = tiling.feat_tile,
        mmq_x = tiling.mmq_x,
        cadence = ?tiling.cadence,
        tiles,
        splits,
        split_launch,
        weight_format = format.kernel_infix,
        path = "mmq_feat_major",
        "CUDA quant kernel: tensor-core MMQ (feature-major)"
    );

    let launch = Launch {
        format,
        client,
        module: &module,
        output_ptr,
        q8_ptr,
        weight_ptr,
        m: m_u32,
        k: k_u32,
        n: n_u32,
        ntok,
        tiling,
        smem,
        grid: (token_tiles, feat_tiles),
        splits,
    };
    if split_launch {
        launch.split_k(device)
    } else {
        launch.tile_parallel()
    }?;

    Ok(Some(()))
}

/// One feature-major launch, on either schedule.
struct Launch<'a> {
    format: &'a FeatMajorFormat,
    client: &'a CudaClient,
    module: &'a std::sync::Arc<CudaModule>,
    output_ptr: u64,
    q8_ptr: u64,
    weight_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    ntok: u32,
    tiling: Tiling,
    smem: u32,
    /// (token tiles, feature tiles).
    grid: (u32, u32),
    splits: u32,
}

impl Launch<'_> {
    fn function(&self, role: Role, smem: u32) -> Result<(CudaFunction, String)> {
        let name = self.tiling.kernel_name(self.format, role);
        let func = kernels::get_kernel_function(self.module, &name)?;
        if smem > 0 {
            opt_in_shared(&func, smem, &name)?;
        }
        Ok((func, name))
    }

    /// One block per output tile. One range takes the plain kernel; more
    /// take the multi-range kernel, which runs them back to back.
    fn tile_parallel(&self) -> Result<()> {
        let role = if self.splits > 1 {
            Role::Fused
        } else {
            Role::TileParallel
        };
        let (func, name) = self.function(role, self.smem)?;
        let cfg = LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, 1),
            block_dim: (self.tiling.threads(), 1, 1),
            shared_mem_bytes: self.smem,
        };
        unsafe {
            let mut builder = self.client.stream().launch_builder(&func);
            builder.arg(&self.q8_ptr);
            builder.arg(&self.weight_ptr);
            builder.arg(&self.output_ptr);
            builder.arg(&self.m);
            builder.arg(&self.k);
            builder.arg(&self.n);
            builder.arg(&self.ntok);
            if self.splits > 1 {
                builder.arg(&self.splits);
            }
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA {name} launch failed: {e:?}"),
            })?;
        }
        Ok(())
    }

    /// One block per (output tile, split range), then the fixup pass that
    /// adds ranges 1.. onto the range-0 store, in order. The fixup is a
    /// separate launch on the same stream: it reads what the first wrote.
    fn split_k(&self, device: &CudaDevice) -> Result<()> {
        let threads = self.tiling.threads();
        let (sk_func, sk_name) = self.function(Role::SplitK, self.smem)?;

        // `workspace[tile][s - 1]`: one dense tile per split past the first.
        let tiles = self.grid.0 as usize * self.grid.1 as usize;
        let ws_len = tiles
            * (self.splits as usize - 1)
            * self.tiling.mmq_x as usize
            * self.tiling.feat_tile as usize;
        let ws = Tensor::<CudaRuntime>::empty(&[ws_len], DType::F32, device)?;
        let ws_ptr = ws.ptr();

        let cfg_sk = LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, self.splits),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: self.smem,
        };
        unsafe {
            let mut builder = self.client.stream().launch_builder(&sk_func);
            builder.arg(&self.q8_ptr);
            builder.arg(&self.weight_ptr);
            builder.arg(&self.output_ptr);
            builder.arg(&ws_ptr);
            builder.arg(&self.m);
            builder.arg(&self.k);
            builder.arg(&self.n);
            builder.arg(&self.ntok);
            builder.launch(cfg_sk).map_err(|e| Error::QuantError {
                reason: format!("CUDA {sk_name} launch failed: {e:?}"),
            })?;
        }

        let (fx_func, fx_name) = self.function(Role::Fixup, 0)?;
        let cfg_fx = LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.client.stream().launch_builder(&fx_func);
            builder.arg(&self.output_ptr);
            builder.arg(&ws_ptr);
            builder.arg(&self.m);
            builder.arg(&self.n);
            builder.arg(&self.splits);
            builder.launch(cfg_fx).map_err(|e| Error::QuantError {
                reason: format!("CUDA {fx_name} launch failed: {e:?}"),
            })?;
        }
        Ok(())
    }
}
