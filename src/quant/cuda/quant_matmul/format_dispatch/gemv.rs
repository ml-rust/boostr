//! GEMV dispatch for CUDA quantized matmul, taken when `m <= gemv_max_m`.

use crate::error::{Error, Result};
use crate::quant::cuda::kernels::{
    self, GEMV_IQ1_M_MODULE, GEMV_IQ1_S_MODULE, GEMV_IQ2_S_MODULE, GEMV_IQ2_XS_MODULE,
    GEMV_IQ2_XXS_MODULE, GEMV_IQ3_S_MODULE, GEMV_IQ3_XXS_MODULE, GEMV_IQ4_NL_MODULE,
    GEMV_IQ4_XS_MODULE, GEMV_Q2_K_MODULE, GEMV_Q3_K_MODULE, GEMV_Q4_1_MODULE, GEMV_Q5_0_MODULE,
    GEMV_Q5_1_MODULE, GEMV_Q5_K_MODULE, GEMV_Q8_1_MODULE, GEMV_Q8_K_MODULE, GEMV_TQ1_0_MODULE,
    GEMV_TQ2_0_MODULE, QUANT_GEMV_MODULE,
};
use crate::quant::cuda::quant_matmul::helpers::quantize_activation_q8_1;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Largest `m` for which the GEMV path beats the feature-major MMQ path.
///
/// GEMV re-reads the whole weight matrix once per token, so its cost scales
/// linearly with `m`. MMQ stages a weight tile once per token tile, so its
/// cost stays flat from `m = 1` up to the tile width. That makes the
/// crossover very low: GEMV only wins while the per-token weight re-read is
/// still cheaper than staging the tile, which for most formats is true only
/// at `m = 1`.
///
/// The values below are measured (see the kernel-comparison example's
/// `--gemv` flag to compare both paths at a given shape) on one GPU
/// architecture and will need re-measuring if either kernel changes or a
/// materially different architecture is targeted.
///
/// The MMQ path needs tensor-core int8 MMA (`caps.int8_mma_m16n8k32`). On a
/// device without it, MMQ isn't available at all, so every format falls back
/// to the old, higher GEMV threshold regardless of format.
pub(in crate::quant::cuda::quant_matmul) fn gemv_max_m(
    format: QuantFormat,
    device_index: usize,
) -> usize {
    let mma_crossover = match format {
        QuantFormat::Q4K => 2,
        // Q8_0's token-batched GEMV covers up to four token columns in one
        // block, so it stays ahead of the feature-major tile further than the
        // per-token kernels do.
        QuantFormat::Q8_0 => 4,
        QuantFormat::Q6K | QuantFormat::Q5K | QuantFormat::Q3K | QuantFormat::Q2K => 1,
        _ => 0,
    };
    let caps = numr::runtime::cuda::CudaDevice::new(device_index)
        .profile()
        .caps;
    if caps.int8_mma_m16n8k32 {
        mma_crossover
    } else {
        16
    }
}

/// GEMV dispatch for M <= 64 (decode + short prefill).
///
/// Chooses the dp4a MWR path for Q4_K / Q6_K / Q8_0 and the F32 activation
/// path for other formats. Returns `Ok(None)` if the format has no dedicated
/// kernel; callers fall back to `quant_matmul_via_dequant`.
pub(in crate::quant::cuda::quant_matmul) fn dispatch_gemv(
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
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // dp4a path: formats with Q8_1 activation + dp4a MWR kernels, aligned K
    if matches!(
        format,
        QuantFormat::Q4K
            | QuantFormat::Q6K
            | QuantFormat::Q8_0
            | QuantFormat::Q5K
            | QuantFormat::Q3K
            | QuantFormat::Q2K
    ) && k.is_multiple_of(32)
    {
        tracing::debug!(
            ?format,
            m,
            k,
            n,
            path = "dp4a_gemv",
            "CUDA quant kernel: dp4a GEMV (optimized)"
        );
        let q8_buf = quantize_activation_q8_1(client, act_contig, m, k)?;
        let q8_ptr = q8_buf.ptr();
        let weight_ptr = weight.storage().ptr();

        let (kernel_name, module_name) = match format {
            QuantFormat::Q4K => ("quant_gemv_q4_k_q8_1_mwr", QUANT_GEMV_MODULE),
            QuantFormat::Q6K => ("quant_gemv_q6_k_q8_1_mwr", QUANT_GEMV_MODULE),
            QuantFormat::Q8_0 => ("quant_gemv_q8_0_q8_1_mwr", QUANT_GEMV_MODULE),
            QuantFormat::Q5K => ("quant_gemv_q5_k_q8_1_mwr", GEMV_Q5_K_MODULE),
            QuantFormat::Q3K => ("quant_gemv_q3_k_q8_1_mwr", GEMV_Q3_K_MODULE),
            QuantFormat::Q2K => ("quant_gemv_q2_k_q8_1_mwr", GEMV_Q2_K_MODULE),
            _ => unreachable!(),
        };

        // Q8_0 has token-batched variants: one block covers `tokens_per_block`
        // token columns, so a weight block is loaded once and dot-producted
        // against all of them. The per-token kernel re-reads the whole weight
        // matrix for every token, which is what makes its cost scale with M.
        // Pick the narrowest tile that covers M in one block — a wider tile
        // would idle its spare columns, a narrower one would need two passes.
        let (kernel_name, tokens_per_block) = match (format, m) {
            (QuantFormat::Q8_0, 0..=1) => (kernel_name, 1),
            (QuantFormat::Q8_0, 2) => ("quant_gemv_q8_0_q8_1_mwr_n2", 2),
            (QuantFormat::Q8_0, _) => ("quant_gemv_q8_0_q8_1_mwr_n4", 4),
            _ => (kernel_name, 1),
        };

        // MWR: one output column per block. The warp count follows the tile
        // width — a wide tile holds one accumulator per token in registers, so
        // it drops to two warps to keep more blocks resident per SM. This
        // mirrors `mwr_nwarps_ntok` in gemv/common.cuh; the two must agree,
        // because the kernel sizes its reduction's shared array from it.
        let block_threads = if tokens_per_block >= 8 { 64 } else { 128 };
        let cfg = LaunchConfig {
            grid_dim: (n_u32, m_u32.div_ceil(tokens_per_block), 1),
            block_dim: (block_threads, 1, 1),
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
        ?format,
        m,
        k,
        n,
        path = "f32_gemv",
        "CUDA quant kernel: F32 GEMV (optimized)"
    );
    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();

    let (kernel_name, module_name) = match format {
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
