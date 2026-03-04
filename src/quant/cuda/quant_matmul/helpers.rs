//! Shared helpers for CUDA quantized matmul operations.

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{self, QUANT_ACT_MODULE};

/// Validate input is F32 and extract (M, K).
pub(super) fn validate_input_cuda(input: &Tensor<CudaRuntime>) -> Result<(usize, usize)> {
    if input.dtype() != DType::F32 {
        return Err(Error::QuantError {
            reason: format!("input must be F32, got {:?}", input.dtype()),
        });
    }
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(Error::QuantError {
            reason: format!("input must be at least 2D, got {:?}", shape),
        });
    }
    let k = shape[shape.len() - 1];
    let m: usize = shape.iter().product::<usize>() / k;
    Ok((m, k))
}

/// Quantize F32 activation to Q8_1 format on GPU.
/// Returns a raw byte tensor of shape [m * num_blocks * 36] containing Q8_1 blocks.
pub(super) fn quantize_activation_q8_1(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    m: usize,
    k: usize,
) -> Result<Tensor<CudaRuntime>> {
    let device_index = activation.device().id();
    let num_blocks = k / 32;
    let q8_bytes = m * num_blocks * 36;

    // Allocate Q8_1 buffer as U8 tensor
    let q8_buf = Tensor::<CudaRuntime>::empty(&[q8_bytes], DType::U8, activation.device());

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_ACT_MODULE)?;
    let func = kernels::get_kernel_function(&module, "quantize_f32_q8_1")?;

    let act_ptr = activation.ptr();
    let q8_ptr = q8_buf.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;

    // Grid: (num_blocks, M, 1), Block: (32, 1, 1) — one warp per Q8_1 block
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, m_u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&q8_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA quantize_f32_q8_1 kernel launch failed: {:?}", e),
        })?;
    }

    Ok(q8_buf)
}
