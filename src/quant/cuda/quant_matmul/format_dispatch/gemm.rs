//! Tiled GEMM dispatch for CUDA quantized matmul, taken when `m` exceeds the
//! per-format GEMV crossover in [`super::gemv::gemv_max_m`].

use crate::error::{Error, Result};
use crate::quant::cuda::kernels::{
    self, GEMM_IQ1_M_MODULE, GEMM_IQ1_S_MODULE, GEMM_IQ2_S_MODULE, GEMM_IQ2_XS_MODULE,
    GEMM_IQ2_XXS_MODULE, GEMM_IQ3_S_MODULE, GEMM_IQ3_XXS_MODULE, GEMM_IQ4_NL_MODULE,
    GEMM_IQ4_XS_MODULE, GEMM_Q2_K_MODULE, GEMM_Q3_K_MODULE, GEMM_Q4_1_MODULE, GEMM_Q5_0_MODULE,
    GEMM_Q5_1_MODULE, GEMM_Q5_K_MODULE, GEMM_Q8_1_MODULE, GEMM_Q8_K_MODULE, GEMM_TQ1_0_MODULE,
    GEMM_TQ2_0_MODULE, QUANT_GEMV_MODULE, QUANT_MATMUL_MODULE, QUANT_MMQ_MMA_MODULE,
};
use crate::quant::cuda::quant_matmul::helpers::quantize_activation_q8_1;
use crate::quant::cuda::quant_matmul::mmq_feat_major;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// The feature-major MMQ descriptor a weight of `format` takes at this `k` on
/// this device, or `None` when the format has no such kernel, `k` is not a
/// whole number of its staging groups, or the device lacks int8 MMA.
///
/// One decision shared by the single-weight GEMM dispatch and the batched
/// path, so the two cannot route the same weight differently. A new format
/// joins by adding a `FeatMajorFormat` and a match arm here.
pub(in crate::quant::cuda::quant_matmul) fn feat_major_format(
    format: QuantFormat,
    k: usize,
    device_index: usize,
) -> Option<&'static mmq_feat_major::FeatMajorFormat> {
    let fm = match format {
        QuantFormat::Q8_0 => &mmq_feat_major::Q8_0,
        QuantFormat::Q4_0 => &mmq_feat_major::Q4_0,
        QuantFormat::Q4_1 => &mmq_feat_major::Q4_1,
        QuantFormat::Q5_0 => &mmq_feat_major::Q5_0,
        QuantFormat::Q5_1 => &mmq_feat_major::Q5_1,
        QuantFormat::Q4K => &mmq_feat_major::Q4_K,
        QuantFormat::Q5K => &mmq_feat_major::Q5_K,
        QuantFormat::Q6K => &mmq_feat_major::Q6_K,
        QuantFormat::Q3K => &mmq_feat_major::Q3_K,
        QuantFormat::Q2K => &mmq_feat_major::Q2_K,
        QuantFormat::IQ4NL => &mmq_feat_major::IQ4_NL,
        QuantFormat::IQ4XS => &mmq_feat_major::IQ4_XS,
        QuantFormat::IQ2XXS => &mmq_feat_major::IQ2_XXS,
        QuantFormat::IQ2XS => &mmq_feat_major::IQ2_XS,
        QuantFormat::IQ2S => &mmq_feat_major::IQ2_S,
        QuantFormat::IQ3XXS => &mmq_feat_major::IQ3_XXS,
        QuantFormat::IQ3S => &mmq_feat_major::IQ3_S,
        QuantFormat::IQ1S => &mmq_feat_major::IQ1_S,
        _ => return None,
    };
    let eligible = k.is_multiple_of(fm.k_multiple as usize)
        && numr::runtime::cuda::CudaDevice::new(device_index)
            .profile()
            .caps
            .int8_mma_m16n8k32;
    eligible.then_some(fm)
}

/// Tiled matmul dispatch for M > 64.
///
/// Returns `Ok(None)` when the format has no dedicated kernel (callers fall
/// back to `quant_matmul_via_dequant`).
pub(in crate::quant::cuda::quant_matmul) fn dispatch_matmul(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();
    let format = weight.format()?;

    let (kernel_name, module_name) = match format {
        QuantFormat::Q4_0 => ("quant_matmul_q4_0_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q8_0 => ("quant_matmul_q8_0_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q4K => ("quant_matmul_q4_k_tiled_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q6K => ("quant_matmul_q6_k_tiled_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q5K => ("quant_matmul_q5_k_f32", GEMM_Q5_K_MODULE),
        QuantFormat::Q3K => ("quant_matmul_q3_k_f32", GEMM_Q3_K_MODULE),
        QuantFormat::Q2K => ("quant_matmul_q2_k_f32", GEMM_Q2_K_MODULE),
        QuantFormat::Q5_0 => ("quant_matmul_q5_0_f32", GEMM_Q5_0_MODULE),
        QuantFormat::IQ4NL => ("quant_matmul_iq4_nl_f32", GEMM_IQ4_NL_MODULE),
        QuantFormat::IQ4XS => ("quant_matmul_iq4_xs_f32", GEMM_IQ4_XS_MODULE),
        QuantFormat::IQ3S => ("quant_matmul_iq3_s_f32", GEMM_IQ3_S_MODULE),
        QuantFormat::IQ2XS => ("quant_matmul_iq2_xs_f32", GEMM_IQ2_XS_MODULE),
        QuantFormat::Q4_1 => ("quant_matmul_q4_1_f32", GEMM_Q4_1_MODULE),
        QuantFormat::Q5_1 => ("quant_matmul_q5_1_f32", GEMM_Q5_1_MODULE),
        QuantFormat::Q8_1 => ("quant_matmul_q8_1_f32", GEMM_Q8_1_MODULE),
        QuantFormat::Q8K => ("quant_matmul_q8_k_f32", GEMM_Q8_K_MODULE),
        QuantFormat::IQ1S => ("quant_matmul_iq1_s_f32", GEMM_IQ1_S_MODULE),
        QuantFormat::IQ1M => ("quant_matmul_iq1_m_f32", GEMM_IQ1_M_MODULE),
        QuantFormat::IQ2XXS => ("quant_matmul_iq2_xxs_f32", GEMM_IQ2_XXS_MODULE),
        QuantFormat::IQ2S => ("quant_matmul_iq2_s_f32", GEMM_IQ2_S_MODULE),
        QuantFormat::IQ3XXS => ("quant_matmul_iq3_xxs_f32", GEMM_IQ3_XXS_MODULE),
        QuantFormat::TQ1_0 => ("quant_matmul_tq1_0_f32", GEMM_TQ1_0_MODULE),
        QuantFormat::TQ2_0 => ("quant_matmul_tq2_0_f32", GEMM_TQ2_0_MODULE),
    };

    tracing::debug!(
        ?format,
        m,
        k,
        n,
        path = "dedicated_gemm",
        "CUDA quant kernel: dedicated tiled GEMM (optimized)"
    );

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL,
    // IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S on sm_80+ take
    // the feature-major tensor-core kernels: a 128-feature tile against a token
    // tile chosen per batch size, with the weight as MMA operand A and a
    // repacked activation layout of its own. It picks between a tile-parallel
    // grid and stream-k internally. `Ok(None)` means no compiled variant fits
    // the device, and the fallback below still serves the shape: the per-format
    // `quant_mmq_*_q8_1_mma` kernel where one exists, and for Q4_0, Q4_1, Q5_0,
    // Q5_1, Q5_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
    // IQ3_S and IQ1_S the dequantize-then-f32 GEMM named by `kernel_name`
    // above, which is the only other GEMM path any of them has. A new format
    // joins by adding a `FeatMajorFormat` and a match arm here.
    if let Some(fm) = feat_major_format(format, k, device_index)
        && mmq_feat_major::dispatch(fm, client, act_contig, weight, output_ptr, m, k, n)?.is_some()
    {
        return Ok(Some(()));
    }

    // Q8_0, Q4_K and Q6_K take the MMQ path: int8 tiles in shared memory and
    // dp4a, the same shape the GEMV's dp4a path uses, but reusing the weight
    // tile across a 128-row batch tile instead of re-reading it per output row.
    // The K-quants additionally need K to be a whole number of 256-element
    // super-blocks, which their own layout already guarantees.
    // All three need sm_80 for their `_mma` kernels (module
    // `QUANT_MMQ_MMA_MODULE`). Below sm_80 each falls back to its dp4a kernel
    // (module `QUANT_GEMV_MODULE`). Both produce bit-identical output.
    // `caps.int8_mma_m16n8k32` names the `m16n8k32` shape, but Q6_K's `_mma`
    // kernel uses `m16n8k16` instead. Both need sm_80 for int8, and neither
    // exists on Turing, so one flag covers both shapes.
    let mmq = match format {
        QuantFormat::Q8_0 if k.is_multiple_of(32) => {
            let caps = numr::runtime::cuda::CudaDevice::new(device_index)
                .profile()
                .caps;
            if caps.int8_mma_m16n8k32 {
                Some(("quant_mmq_q8_0_q8_1_mma", QUANT_MMQ_MMA_MODULE))
            } else {
                Some(("quant_mmq_q8_0_q8_1", QUANT_GEMV_MODULE))
            }
        }
        QuantFormat::Q4K if k.is_multiple_of(256) => {
            let caps = numr::runtime::cuda::CudaDevice::new(device_index)
                .profile()
                .caps;
            if caps.int8_mma_m16n8k32 {
                Some(("quant_mmq_q4_k_q8_1_mma", QUANT_MMQ_MMA_MODULE))
            } else {
                Some(("quant_mmq_q4_k_q8_1", QUANT_GEMV_MODULE))
            }
        }
        QuantFormat::Q6K if k.is_multiple_of(256) => {
            let caps = numr::runtime::cuda::CudaDevice::new(device_index)
                .profile()
                .caps;
            if caps.int8_mma_m16n8k32 {
                Some(("quant_mmq_q6_k_q8_1_mma", QUANT_MMQ_MMA_MODULE))
            } else {
                Some(("quant_mmq_q6_k_q8_1", QUANT_GEMV_MODULE))
            }
        }
        _ => None,
    };
    if let Some((mmq_kernel, mmq_module)) = mmq {
        let q8_buf = quantize_activation_q8_1(client, act_contig, m, k)?;
        let q8_ptr = q8_buf.ptr();
        let weight_ptr = weight.storage().ptr();
        let cfg = LaunchConfig {
            grid_dim: (n_u32.div_ceil(64), m_u32.div_ceil(128), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let module = kernels::get_or_load_module(client.context(), device_index, mmq_module)?;
        let func = kernels::get_kernel_function(&module, mmq_kernel)?;
        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&q8_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&output_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA {mmq_kernel} launch failed: {:?}", e),
            })?;
        }
        return Ok(Some(()));
    }

    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();

    // Q8_0, Q4_K and Q6_K share one 64×64-tile kernel with 256 threads, each
    // holding a 4×4 register patch and both operands staged in shared memory.
    // The weight is then read once per BLOCK instead of once per OUTPUT, which
    // is what stops the cost scaling with the batch; see the kernel comment.
    // Shared memory is declared statically inside it, so none is requested here.
    // Every other format still uses the classic per-element 16×16 GEMM.
    let cfg = if matches!(
        format,
        QuantFormat::Q8_0 | QuantFormat::Q4K | QuantFormat::Q6K
    ) {
        LaunchConfig {
            grid_dim: (n_u32.div_ceil(64), m_u32.div_ceil(64), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        let block_x = 16u32;
        let block_y = 16u32;
        LaunchConfig {
            grid_dim: (n_u32.div_ceil(block_x), m_u32.div_ceil(block_y), 1),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        }
    };

    let module = kernels::get_or_load_module(client.context(), device_index, module_name)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA quant_matmul kernel launch failed: {:?}", e),
        })?;
    }

    Ok(Some(()))
}
