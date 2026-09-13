//! `quant_matmul` through the feature-major MMQ path at a caller-chosen
//! feature tile. Exists ONLY for A/B measurement in
//! `examples/quant_shape_bench.rs`, so the two feature tiles can be timed at
//! one shape. NOT a production API: production code takes
//! `QuantMatmulOps::quant_matmul`, whose tile rule is automatic.

use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::format_dispatch::feat_major_format;
use super::helpers::validate_input_cuda;
use super::mmq_feat_major::{self, FeatTile};

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
