//! The single-token schedule of the feature-major family: `M = 1` on the
//! formats `quant_mmq_gemv1.cu` compiles, which are the formats
//! `mmqf_vec_dot_d` serves as a Q8_0 row. Same activation record, same K
//! ranges (`split_count`), same float sequence per output element as the
//! tensor-core kernels, so a decode step is the same bits as row 0 of any
//! batch. A third schedule beside the tile-parallel grid and the split-K
//! pair; the dispatch takes it under [`Schedule::Auto`] alone, so the forced
//! schedules still reach the tensor-core kernels for comparison.

use crate::error::{Error, Result};
use crate::quant::QuantFormat;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use super::super::super::kernels::{self, QUANT_MMQ_GEMV1_MODULE};
use super::formats::FeatMajorFormat;
use super::tiling::{FeatTile, Schedule, split_count};

/// Output features per block: `GEMV1_FEATS_PER_BLOCK` in `mmq/gemv1_body.cuh`.
const FEATS_PER_BLOCK: u32 = 32;
/// Threads per block: `GEMV1_THREADS` in `mmq/gemv1_body.cuh`.
const THREADS: u32 = 128;
/// Threads per fixup block: `GEMV1_FIXUP_THREADS` in `mmq/gemv1_body.cuh`.
const FIXUP_THREADS: u32 = 256;
/// The split-range fixup, shared by every format.
const FIXUP_KERNEL: &str = "quant_mmq_q8_1_gemv1_fixup";

/// The `M = 1` kernel of `format`, `None` for a format without one. The
/// K-quants, the 4- and 5-bit legacy formats and the IQ formats stay on the
/// tensor-core kernels at every M. The name follows `format.kernel_infix`,
/// the same field `Tiling::kernel_name` reads for the tensor-core variants.
pub(super) fn kernel_name(format: &FeatMajorFormat) -> Option<String> {
    match format.quant_format {
        QuantFormat::Q8_0
        | QuantFormat::PQ2_0
        | QuantFormat::Q2_0
        | QuantFormat::Q1_0
        | QuantFormat::PTQ1_0 => Some(format!("quant_mmq_{}_q8_1_gemv1", format.kernel_infix)),
        _ => None,
    }
}

/// The kernel this call takes instead of a tensor-core variant: one token,
/// a format with an `M = 1` kernel, and neither the tile nor the schedule
/// forced. `dispatch` and `dispatch_quantized` both ask, so the record the
/// first quantizes is the one the second launches.
pub(super) fn serves(
    format: &FeatMajorFormat,
    m: usize,
    feat_tile: FeatTile,
    schedule: Schedule,
) -> Option<String> {
    if m != 1 || feat_tile != FeatTile::Auto || schedule != Schedule::Auto {
        return None;
    }
    kernel_name(format)
}

/// Launches `name` over `[1, K] x [N, K]^T` on the activation record at
/// `q8_ptr` with token stride `ntok`, then the fixup when the split count is
/// above one. `splits` is `split_count(k, n, sms)`, the count the
/// tensor-core launch would use at this K and N.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    client: &CudaClient,
    device: &CudaDevice,
    name: &str,
    q8_ptr: u64,
    ntok: u32,
    weight_ptr: u64,
    output_ptr: u64,
    k: u32,
    n: u32,
    sms: u32,
) -> Result<()> {
    let splits = split_count(k, n, sms);
    let module =
        kernels::get_or_load_module(client.context(), device.id(), QUANT_MMQ_GEMV1_MODULE)?;
    let func = kernels::get_kernel_function(&module, name)?;

    // `workspace[s - 1][N]` for every range past the first; one range
    // reads none, and a one-element buffer keeps the argument a live
    // allocation.
    let ws_len = (splits as usize - 1) * n as usize;
    let ws = Tensor::<CudaRuntime>::empty(&[ws_len.max(1)], DType::F32, device)?;
    let ws_ptr = ws.ptr();

    tracing::debug!(
        k,
        n,
        splits,
        kernel = name,
        path = "mmq_feat_major",
        "CUDA quant kernel: single-token MMQ (feature-major)"
    );

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(FEATS_PER_BLOCK), splits, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q8_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&ws_ptr);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&ntok);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA {name} launch failed: {e:?}"),
        })?;
    }
    if splits < 2 {
        return Ok(());
    }

    // A separate launch on the same stream: it reads what the first wrote.
    let fx_func = kernels::get_kernel_function(&module, FIXUP_KERNEL)?;
    let cfg_fx = LaunchConfig {
        grid_dim: (n.div_ceil(FIXUP_THREADS), 1, 1),
        block_dim: (FIXUP_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&fx_func);
        builder.arg(&output_ptr);
        builder.arg(&ws_ptr);
        builder.arg(&n);
        builder.arg(&splits);
        builder.launch(cfg_fx).map_err(|e| Error::QuantError {
            reason: format!("CUDA {FIXUP_KERNEL} launch failed: {e:?}"),
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::formats::{IQ4_NL, PQ2_0, PTQ1_0, Q1_0, Q2_0, Q4_0, Q4_K, Q8_0};
    use super::*;

    /// The five formats `mmqf_vec_dot_d` serves as a Q8_0 row have the
    /// kernel and are named by the `MMQ_GEMV1_KERNEL` spelling; the rest
    /// stay on the tensor-core kernels.
    #[test]
    fn the_kernel_exists_for_the_q8_0_row_formats_alone() {
        for (fm, infix) in [
            (&Q8_0, "q8_0"),
            (&PQ2_0, "pq2_0"),
            (&Q2_0, "q2_0"),
            (&Q1_0, "q1_0"),
            (&PTQ1_0, "ptq1_0"),
        ] {
            assert_eq!(
                kernel_name(fm),
                Some(format!("quant_mmq_{infix}_q8_1_gemv1"))
            );
        }
        assert_eq!(kernel_name(&Q4_0), None);
        assert_eq!(kernel_name(&Q4_K), None);
        assert_eq!(kernel_name(&IQ4_NL), None);
    }

    /// Only the automatic tile and schedule at one token take it.
    #[test]
    fn only_one_token_on_the_automatic_schedule_takes_it() {
        assert!(serves(&Q8_0, 1, FeatTile::Auto, Schedule::Auto).is_some());
        assert!(serves(&Q8_0, 2, FeatTile::Auto, Schedule::Auto).is_none());
        assert!(serves(&Q8_0, 1, FeatTile::Force(16), Schedule::Auto).is_none());
        assert!(serves(&Q8_0, 1, FeatTile::Auto, Schedule::TileParallel).is_none());
        assert!(serves(&Q8_0, 1, FeatTile::Auto, Schedule::SplitK).is_none());
        assert!(serves(&Q4_K, 1, FeatTile::Auto, Schedule::Auto).is_none());
    }
}
