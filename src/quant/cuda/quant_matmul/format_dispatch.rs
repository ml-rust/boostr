//! Format-specific dispatch for CUDA quantized GEMV and tiled GEMM.
//!
//! - `dispatch_gemv`   — GEMV path (M <= 64) for `quant_matmul`
//! - `dispatch_matmul` — tiled matmul path (M > 64) for `quant_matmul`

use crate::error::{Error, Result};
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{
    self, GEMM_IQ1_M_MODULE, GEMM_IQ1_S_MODULE, GEMM_IQ2_S_MODULE, GEMM_IQ2_XS_MODULE,
    GEMM_IQ2_XXS_MODULE, GEMM_IQ3_S_MODULE, GEMM_IQ3_XXS_MODULE, GEMM_IQ4_NL_MODULE,
    GEMM_IQ4_XS_MODULE, GEMM_Q2_K_MODULE, GEMM_Q3_K_MODULE, GEMM_Q4_1_MODULE, GEMM_Q5_0_MODULE,
    GEMM_Q5_1_MODULE, GEMM_Q5_K_MODULE, GEMM_Q8_1_MODULE, GEMM_Q8_K_MODULE, GEMM_TQ1_0_MODULE,
    GEMM_TQ2_0_MODULE, GEMV_IQ1_M_MODULE, GEMV_IQ1_S_MODULE, GEMV_IQ2_S_MODULE, GEMV_IQ2_XS_MODULE,
    GEMV_IQ2_XXS_MODULE, GEMV_IQ3_S_MODULE, GEMV_IQ3_XXS_MODULE, GEMV_IQ4_NL_MODULE,
    GEMV_IQ4_XS_MODULE, GEMV_Q2_K_MODULE, GEMV_Q3_K_MODULE, GEMV_Q4_1_MODULE, GEMV_Q5_0_MODULE,
    GEMV_Q5_1_MODULE, GEMV_Q5_K_MODULE, GEMV_Q8_1_MODULE, GEMV_Q8_K_MODULE, GEMV_TQ1_0_MODULE,
    GEMV_TQ2_0_MODULE, QUANT_GEMV_MODULE, QUANT_MATMUL_MODULE,
};
use super::helpers::quantize_activation_q8_1;

/// GEMV dispatch for M <= 64 (decode + short prefill).
///
/// Chooses the dp4a MWR path for Q4_K / Q6_K / Q8_0 and the F32 activation
/// path for other formats. Returns `Ok(None)` if the format has no dedicated
/// kernel (callers should then fall back to `quant_matmul_via_dequant`).
pub(super) fn dispatch_gemv(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // dp4a path: formats with Q8_1 activation + dp4a MWR kernels, aligned K
    if matches!(
        weight.format(),
        QuantFormat::Q4K
            | QuantFormat::Q6K
            | QuantFormat::Q8_0
            | QuantFormat::Q5K
            | QuantFormat::Q3K
            | QuantFormat::Q2K
    ) && k % 32 == 0
    {
        tracing::debug!(
            format = ?weight.format(),
            m, k, n,
            path = "dp4a_gemv",
            "CUDA quant kernel: dp4a GEMV (optimized)"
        );
        let q8_buf = quantize_activation_q8_1(client, act_contig, m, k)?;
        let q8_ptr = q8_buf.ptr();
        let weight_ptr = weight.storage().ptr();

        let (kernel_name, module_name) = match weight.format() {
            QuantFormat::Q4K => ("quant_gemv_q4_k_q8_1_mwr", QUANT_GEMV_MODULE),
            QuantFormat::Q6K => ("quant_gemv_q6_k_q8_1_mwr", QUANT_GEMV_MODULE),
            QuantFormat::Q8_0 => ("quant_gemv_q8_0_q8_1_mwr", QUANT_GEMV_MODULE),
            QuantFormat::Q5K => ("quant_gemv_q5_k_q8_1_mwr", GEMV_Q5_K_MODULE),
            QuantFormat::Q3K => ("quant_gemv_q3_k_q8_1_mwr", GEMV_Q3_K_MODULE),
            QuantFormat::Q2K => ("quant_gemv_q2_k_q8_1_mwr", GEMV_Q2_K_MODULE),
            _ => unreachable!(),
        };

        // MWR: one output column per block, 128 threads (4 warps)
        let cfg = LaunchConfig {
            grid_dim: (n_u32, m_u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = kernels::get_or_load_module(client.context(), device_index, module_name)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&q8_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&output_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA {} launch failed: {:?}", kernel_name, e),
            })?;
        }
        return Ok(Some(()));
    }

    // F32 activation path for formats with dedicated F32 GEMV kernels
    tracing::debug!(
        format = ?weight.format(),
        m, k, n,
        path = "f32_gemv",
        "CUDA quant kernel: F32 GEMV (optimized)"
    );
    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();

    let (kernel_name, module_name) = match weight.format() {
        QuantFormat::Q4_0 => ("quant_gemv_q4_0_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q8_0 => ("quant_gemv_q8_0_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q4K => ("quant_gemv_q4_k_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q6K => ("quant_gemv_q6_k_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q5K => ("quant_gemv_q5_k_f32", GEMV_Q5_K_MODULE),
        QuantFormat::Q3K => ("quant_gemv_q3_k_f32", GEMV_Q3_K_MODULE),
        QuantFormat::Q2K => ("quant_gemv_q2_k_f32", GEMV_Q2_K_MODULE),
        QuantFormat::Q5_0 => ("quant_gemv_q5_0_f32", GEMV_Q5_0_MODULE),
        QuantFormat::IQ4NL => ("quant_gemv_iq4_nl_f32", GEMV_IQ4_NL_MODULE),
        QuantFormat::IQ4XS => ("quant_gemv_iq4_xs_f32", GEMV_IQ4_XS_MODULE),
        QuantFormat::IQ3S => ("quant_gemv_iq3_s_f32", GEMV_IQ3_S_MODULE),
        QuantFormat::IQ2XS => ("quant_gemv_iq2_xs_f32", GEMV_IQ2_XS_MODULE),
        QuantFormat::Q4_1 => ("quant_gemv_q4_1_f32", GEMV_Q4_1_MODULE),
        QuantFormat::Q5_1 => ("quant_gemv_q5_1_f32", GEMV_Q5_1_MODULE),
        QuantFormat::Q8_1 => ("quant_gemv_q8_1_f32", GEMV_Q8_1_MODULE),
        QuantFormat::Q8K => ("quant_gemv_q8_k_f32", GEMV_Q8_K_MODULE),
        QuantFormat::IQ1S => ("quant_gemv_iq1_s_f32", GEMV_IQ1_S_MODULE),
        QuantFormat::IQ1M => ("quant_gemv_iq1_m_f32", GEMV_IQ1_M_MODULE),
        QuantFormat::IQ2XXS => ("quant_gemv_iq2_xxs_f32", GEMV_IQ2_XXS_MODULE),
        QuantFormat::IQ2S => ("quant_gemv_iq2_s_f32", GEMV_IQ2_S_MODULE),
        QuantFormat::IQ3XXS => ("quant_gemv_iq3_xxs_f32", GEMV_IQ3_XXS_MODULE),
        QuantFormat::TQ1_0 => ("quant_gemv_tq1_0_f32", GEMV_TQ1_0_MODULE),
        QuantFormat::TQ2_0 => ("quant_gemv_tq2_0_f32", GEMV_TQ2_0_MODULE),
    };

    let warps_per_block = 8u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(warps_per_block), m_u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
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
            reason: format!("CUDA quant_gemv kernel launch failed: {:?}", e),
        })?;
    }

    Ok(Some(()))
}

/// Tiled matmul dispatch for M > 64.
///
/// Returns `Ok(None)` when the format has no dedicated kernel (callers fall
/// back to `quant_matmul_via_dequant`).
pub(super) fn dispatch_matmul(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();

    let (kernel_name, module_name) = match weight.format() {
        QuantFormat::Q4_0 => ("quant_matmul_q4_0_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q8_0 => ("quant_matmul_q8_0_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q4K => ("quant_matmul_q4_k_f32", QUANT_MATMUL_MODULE),
        QuantFormat::Q6K => ("quant_matmul_q6_k_f32", QUANT_MATMUL_MODULE),
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
        format = ?weight.format(),
        m, k, n,
        path = "dedicated_gemm",
        "CUDA quant kernel: dedicated tiled GEMM (optimized)"
    );

    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let block_x = 16u32;
    let block_y = 16u32;
    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(block_x), m_u32.div_ceil(block_y), 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
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
