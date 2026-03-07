//! Generic fallback kernels for CUDA quantized matmul.
//!
//! - `quant_matmul_via_dequant` — fused dequant+dot fallback (all quant formats)
//! - `quant_swiglu_via_dequant` — fused SwiGLU fallback (all quant formats)

use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{self, QUANT_MATMUL_GENERIC_MODULE};

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
