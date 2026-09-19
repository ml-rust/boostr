//! `quant_matmul` through the feature-major MMQ path at a caller-chosen
//! feature tile or launch schedule, plus the tile-parallel probe's inputs.
//! Exists ONLY for measurement: the kernel A/B in
//! `examples/quant_shape_bench.rs` and the schedule invariance test in
//! `tests/quant_mmq_tile_parallel_tune.rs`. NOT a production API: production
//! code takes `QuantMatmulOps::quant_matmul`, whose tile and schedule rules
//! are automatic.

use crate::error::{Error, Result};
use crate::quant::{QuantFormat, QuantTensor};
use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::runtime::{Device, RuntimeClient};
use numr::tensor::Tensor;

use super::format_dispatch::feat_major_format;
use super::helpers::validate_input_cuda;
use super::mmq_feat_major::{self, FeatMajorFormat, FeatTile, Schedule};

/// Runs `activation x weight^T` on the feature-major MMQ kernels at
/// `feat_tile`, skipping the GEMV crossover so the kernel under test runs at
/// every `m`. Errors when the format has no feature-major kernel at this
/// shape, when no compiled variant fits the device, or when `feat_tile`
/// forces a tile the format does not compile.
#[doc(hidden)]
pub fn quant_matmul_forced_feat_tile(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    feat_tile: FeatTile,
) -> Result<Tensor<CudaRuntime>> {
    forced(client, activation, weight, feat_tile, Schedule::Auto)
}

/// Runs `activation x weight^T` on the feature-major MMQ kernels at the
/// automatic feature tile and the forced `schedule`. Errors as
/// [`quant_matmul_forced_feat_tile`] does, and when `Schedule::SplitK` is
/// forced at a shape whose split count is 1.
#[doc(hidden)]
pub fn quant_matmul_forced_schedule(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    schedule: Schedule,
) -> Result<Tensor<CudaRuntime>> {
    forced(client, activation, weight, FeatTile::Auto, schedule)
}

/// The measured `prefers_tile_parallel` pick for `format` on `client`'s
/// device: the value the dispatch reads. The first call per (device, format)
/// runs the probe; with `NUMR_CUDA_TUNE=0` it is the descriptor's fallback.
#[doc(hidden)]
pub fn mmq_prefers_tile_parallel(client: &CudaClient, format: QuantFormat) -> Result<bool> {
    Ok(mmq_feat_major::prefers_tile_parallel(
        client,
        descriptor(client, format)?,
    ))
}

/// The `(m, n, k)` the tile-parallel probe times for `format` on `client`'s
/// device.
#[doc(hidden)]
pub fn mmq_tile_parallel_probe_shape(
    client: &CudaClient,
    format: QuantFormat,
) -> Result<(usize, usize, usize)> {
    let (m, n, k) = mmq_feat_major::tile_parallel_probe_shape(client, descriptor(client, format)?)?;
    Ok((m as usize, n as usize, k as usize))
}

/// The feature-major descriptor of `format` on `client`'s device. `k = 0`
/// is a whole number of every block, so only the format and the device's
/// int8 MMA gate the lookup.
fn descriptor(client: &CudaClient, format: QuantFormat) -> Result<&'static FeatMajorFormat> {
    feat_major_format(format, 0, client.device().id()).ok_or_else(|| Error::QuantError {
        reason: format!(
            "{} has no feature-major MMQ kernel on this device",
            format.name()
        ),
    })
}

fn forced(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    feat_tile: FeatTile,
    schedule: Schedule,
) -> Result<Tensor<CudaRuntime>> {
    let (m, k) = validate_input_cuda(activation)?;
    let w_shape = weight.shape();
    if w_shape.len() != 2 || w_shape[1] != k {
        return Err(Error::QuantError {
            reason: format!(
                "quant_matmul_forced_feat_tile weight shape {w_shape:?} does not match [N, {k}]"
            ),
        });
    }
    let n = w_shape[0];
    let format = weight.format();
    let device_index = activation.device().id();
    let Some(fm) = feat_major_format(format, k, device_index) else {
        return Err(Error::QuantError {
            reason: format!(
                "quant_matmul_forced_feat_tile: {} has no feature-major MMQ kernel at K={k} on this device; use a format and K the path serves",
                format.name()
            ),
        });
    };

    let act_contig = activation.contiguous()?;
    let a_shape = activation.shape();
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(n);
    let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device())?;

    let launched = mmq_feat_major::dispatch(
        fm,
        client,
        &act_contig,
        weight,
        output.ptr(),
        m,
        k,
        n,
        feat_tile,
        schedule,
    )?;
    if launched.is_none() {
        return Err(Error::QuantError {
            reason: format!(
                "quant_matmul_forced_feat_tile: no compiled {} variant fits {m} tokens at feature tile {feat_tile:?} on this device",
                format.name()
            ),
        });
    }
    Ok(output)
}
