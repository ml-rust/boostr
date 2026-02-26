//! CUDA INT4 GEMM dispatch helpers (AWQ, GPTQ, Marlin)

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, INT4_GEMM_GPTQ_MODULE, INT4_GEMM_MODULE, MARLIN_GEMM_MODULE};

#[allow(clippy::too_many_arguments)]
pub fn launch_int4_gemm(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    qweight: &Tensor<CudaRuntime>,
    scales: &Tensor<CudaRuntime>,
    zeros: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
) -> Result<()> {
    let device_index = input.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, INT4_GEMM_MODULE)?;
    let func = kernels::get_kernel_function(&module, "int4_gemm_f32")?;

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(16), m.div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let input_ptr = input.ptr();
    let qweight_ptr = qweight.ptr();
    let scales_ptr = scales.ptr();
    let zeros_ptr = zeros.ptr();
    let output_ptr = output.ptr();

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&qweight_ptr);
        builder.arg(&scales_ptr);
        builder.arg(&zeros_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&group_size);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA int4_gemm launch failed: {:?}", e),
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn launch_int4_gemm_gptq(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    qweight: &Tensor<CudaRuntime>,
    qzeros: &Tensor<CudaRuntime>,
    scales: &Tensor<CudaRuntime>,
    g_idx: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    let device_index = input.device().id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, INT4_GEMM_GPTQ_MODULE)?;
    let func = kernels::get_kernel_function(&module, "int4_gemm_gptq_f32")?;

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(16), m.div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let input_ptr = input.ptr();
    let qweight_ptr = qweight.ptr();
    let qzeros_ptr = qzeros.ptr();
    let scales_ptr = scales.ptr();
    let g_idx_ptr = g_idx.ptr();
    let output_ptr = output.ptr();

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&qweight_ptr);
        builder.arg(&qzeros_ptr);
        builder.arg(&scales_ptr);
        builder.arg(&g_idx_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA int4_gemm_gptq launch failed: {:?}", e),
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn launch_marlin_gemm(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    scales: &Tensor<CudaRuntime>,
    zeros: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
) -> Result<()> {
    let device_index = input.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, MARLIN_GEMM_MODULE)?;
    let func = kernels::get_kernel_function(&module, "marlin_gemm_f32")?;

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(16), m.div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let input_ptr = input.ptr();
    let weight_ptr = weight.ptr();
    let scales_ptr = scales.ptr();
    let zeros_ptr = zeros.ptr();
    let output_ptr = output.ptr();

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&scales_ptr);
        builder.arg(&zeros_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&group_size);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA marlin_gemm launch failed: {:?}", e),
        })?;
    }
    Ok(())
}
