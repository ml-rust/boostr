//! Generic fallback and dispatch helpers for CUDA quantized matmul.
//!
//! - `quant_matmul_via_dequant` — fused dequant+dot fallback (all quant formats)
//! - `quant_swiglu_via_dequant` — fused SwiGLU fallback (all quant formats)
//! - `dispatch_gemv`            — GEMV path (M <= 64) for `quant_matmul`
//! - `dispatch_matmul`          — tiled matmul path (M > 64) for `quant_matmul`

use crate::error::{Error, Result};
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{
    self, QUANT_GEMV_MODULE, QUANT_MATMUL_GENERIC_MODULE, QUANT_MATMUL_MODULE,
};
use super::helpers::quantize_activation_q8_1;

// ---------------------------------------------------------------------------
// Generic fallbacks (dequant-on-the-fly)
// ---------------------------------------------------------------------------

/// Generic fallback: fused dequant+dot CUDA kernel for all quant formats.
/// Dequantizes weight blocks in registers during matmul — never materializes full f32 weight.
/// Used for quant formats without dedicated CUDA GEMM/GEMV kernels.
pub(super) fn quant_matmul_via_dequant(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let a_shape = activation.shape();
    let w_shape = weight.shape();
    let n = w_shape[0];
    let k = w_shape[1];
    let total: usize = a_shape.iter().product();
    let m = total / k;

    let act_contig = activation.contiguous();
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(n);
    let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());

    let device_index = activation.device().id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MATMUL_GENERIC_MODULE)?;
    let func = kernels::get_kernel_function(&module, "quant_matmul_generic_f32")?;

    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();
    let output_ptr = output.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;
    let format_id = weight.format().format_id();

    // Grid: (N, M, 1), Block: (32, 1, 1) — one warp per output element
    let cfg = LaunchConfig {
        grid_dim: (n_u32, m_u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.arg(&format_id);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA quant_matmul_generic_f32 launch failed: {:?}", e),
        })?;
    }

    Ok(output)
}

/// Generic fused SwiGLU: gate_matmul + up_matmul + silu(gate)*up in one kernel.
/// Eliminates 2 intermediate tensors and reduces kernel launches from 3 to 1.
pub(super) fn quant_swiglu_via_dequant(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    gate_weight: &QuantTensor<CudaRuntime>,
    up_weight: &QuantTensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let device_index = activation.device().id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MATMUL_GENERIC_MODULE)?;
    let func = kernels::get_kernel_function(&module, "quant_swiglu_generic_f32")?;

    let act_ptr = activation.ptr();
    let gate_ptr = gate_weight.storage().ptr();
    let up_ptr = up_weight.storage().ptr();
    let output_ptr = output.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;
    let format_id = gate_weight.format().format_id();

    let cfg = LaunchConfig {
        grid_dim: (n_u32, m_u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&gate_ptr);
        builder.arg(&up_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.arg(&format_id);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA quant_swiglu_generic_f32 launch failed: {:?}", e),
        })?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch helpers for quant_matmul (extracted to keep impl_ops.rs lean)
// ---------------------------------------------------------------------------

/// GEMV dispatch for M <= 64 (decode + short prefill).
///
/// Chooses the dp4a MWR path for Q4_K / Q6_K / Q8_0 and the F32 activation
/// path for other formats. Returns `Err` if the format has no dedicated kernel
/// (callers should then fall back to `quant_matmul_via_dequant`).
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

    // dp4a path: Q4_K / Q6_K / Q8_0 + aligned K
    if matches!(
        weight.format(),
        QuantFormat::Q4K | QuantFormat::Q6K | QuantFormat::Q8_0
    ) && k % 32 == 0
    {
        let q8_buf = quantize_activation_q8_1(client, act_contig, m, k)?;
        let q8_ptr = q8_buf.ptr();
        let weight_ptr = weight.storage().ptr();

        let kernel_name = match weight.format() {
            QuantFormat::Q4K => "quant_gemv_q4_k_q8_1_mwr",
            QuantFormat::Q6K => "quant_gemv_q6_k_q8_1_mwr",
            QuantFormat::Q8_0 => "quant_gemv_q8_0_q8_1_mwr",
            _ => unreachable!(),
        };

        // MWR: one output column per block, 128 threads (4 warps)
        let cfg = LaunchConfig {
            grid_dim: (n_u32, m_u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let module =
            kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE)?;
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

    // F32 activation path for other formats
    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();

    let kernel_name = match weight.format() {
        QuantFormat::Q4_0 => "quant_gemv_q4_0_f32",
        QuantFormat::Q8_0 => "quant_gemv_q8_0_f32",
        QuantFormat::Q4K => "quant_gemv_q4_k_f32",
        QuantFormat::Q6K => "quant_gemv_q6_k_f32",
        _ => return Ok(None), // Signal: caller should use dequant fallback
    };

    let warps_per_block = 8u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(warps_per_block), m_u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE)?;
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

    let kernel_name = match weight.format() {
        QuantFormat::Q4_0 => "quant_matmul_q4_0_f32",
        QuantFormat::Q8_0 => "quant_matmul_q8_0_f32",
        QuantFormat::Q4K => "quant_matmul_q4_k_f32",
        QuantFormat::Q6K => "quant_matmul_q6_k_f32",
        _ => return Ok(None), // Signal: caller should use dequant fallback
    };

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

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_MATMUL_MODULE)?;
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
