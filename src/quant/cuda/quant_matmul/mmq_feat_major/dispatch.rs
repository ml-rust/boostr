use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use super::super::super::kernels::{self, QUANT_MMQ_MMA_MODULE};
use super::super::helpers::quantize_activation_q8_1_mmq;
use super::formats::FeatMajorFormat;
use super::gemv1;
use super::launch::Launch;
use super::tiling::{
    Cadence, FEAT_TILE_DEFAULT, FeatTile, Schedule, Tiling, prefers_tile_parallel,
    record_token_slots, select_tiling, select_variant, smem_opt_in_limit, split_count,
    use_split_launch,
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
/// covers the tile the second launches. `prefers_tile_parallel` is the
/// format's measured pick on this device.
fn tiling_for(
    format: &FeatMajorFormat,
    profile_index: usize,
    m: usize,
    n: usize,
    k: usize,
    feat_tile: FeatTile,
    prefers_tile_parallel: bool,
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
        prefers_tile_parallel,
    )
}

/// Runs one quantized weight x F32 activation through the feature-major MMQ
/// kernels. `Ok(None)` means no variant fits, and the caller should keep its
/// existing path.
///
/// `feat_tile` is [`FeatTile::Auto`] and `schedule` is [`Schedule::Auto`]
/// for every production caller; a forced tile or schedule is a measurement
/// hook and errors when the format or shape lacks it. At one token on a
/// format with an `M = 1` kernel (`gemv1::serves`) the automatic schedule
/// launches that kernel, which forms the same bits as the tensor-core
/// variant it stands in for.
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
    schedule: Schedule,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();
    let prefers = prefers_tile_parallel(client, format);
    let Some(tiling) = tiling_for(format, device_index, m, n, k, feat_tile, prefers)? else {
        return Ok(None);
    };

    // This path reads its own activation layout, k-group-major and
    // token-minor, so the tile copy is flat. The per-token producer stays
    // untouched for `quant_mmq_q8_0_q8_1_mma`, dp4a and the K-quants. The
    // single-token kernel reads one record per k-group, so its token slot
    // count is one.
    let slots = if gemv1::serves(format, m, feat_tile, schedule).is_some() {
        1
    } else {
        tiling.mmq_x as usize
    };
    let (q8_buf, ntok) = quantize_activation_q8_1_mmq(client, act_contig, m, k, slots)?;
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
        schedule,
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
    schedule: Schedule,
) -> Result<Option<()>> {
    let device_index = device.id();
    let profile = CudaDevice::new(device_index).profile();

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // Measured once per (device, format) and cached; the first call on a
    // device runs the probe in `tiling::tile_parallel_tune`.
    let prefers = prefers_tile_parallel(client, format);
    let Some(tiling) = tiling_for(format, device_index, m, n, k, feat_tile, prefers)? else {
        return Ok(None);
    };
    let weight_ptr = weight.storage().ptr();
    let sms = profile.compute_units;

    // The single-token schedule: the tensor-core variant fits, so the M = 1
    // kernel stands in for it with the same split count and the same bits.
    if let Some(name) = gemv1::serves(format, m, feat_tile, schedule) {
        gemv1::launch(
            client, device, &name, q8_ptr, ntok, weight_ptr, output_ptr, k_u32, n_u32, sms,
        )?;
        return Ok(Some(()));
    }

    let smem = tiling.smem_bytes(format);

    let token_tiles = tiling.token_tiles(m_u32);
    let feat_tiles = tiling.feat_tiles(n_u32);
    let tiles = token_tiles * feat_tiles;

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

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)?;

    // The split count reads K, N and the device only, so the float sequence
    // each output element receives is fixed before M or the tiling is known.
    let splits = split_count(k_u32, n_u32, sms);
    let split_launch = match schedule {
        Schedule::Auto => use_split_launch(splits, tiles, tiling.feat_tile, sms, prefers),
        Schedule::SplitK if splits > 1 => true,
        Schedule::SplitK => {
            return Err(Error::QuantError {
                reason: format!(
                    "MMQ split-K schedule forced at K={k} N={n}, where the split count is 1; \
                     the pair exists from K=2048 up with fewer feature tiles than SMs"
                ),
            });
        }
        Schedule::TileParallel => false,
    };

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
