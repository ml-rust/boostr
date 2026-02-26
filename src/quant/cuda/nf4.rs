//! CUDA NF4 dispatch helpers

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, NF4_QUANT_MODULE};

pub fn launch_nf4_dequant(
    client: &CudaClient,
    nf4_data: &Tensor<CudaRuntime>,
    absmax: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    num_bytes: u32,
    blocksize: u32,
) -> Result<()> {
    let device_index = nf4_data.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, NF4_QUANT_MODULE)?;
    let func = kernels::get_kernel_function(&module, "nf4_dequant_f32")?;

    let block_size = 256u32;
    let grid_size = (num_bytes + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let nf4_ptr = nf4_data.ptr();
    let absmax_ptr = absmax.ptr();
    let output_ptr = output.ptr();

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&nf4_ptr);
        builder.arg(&absmax_ptr);
        builder.arg(&output_ptr);
        builder.arg(&num_bytes);
        builder.arg(&blocksize);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA nf4_dequant launch failed: {:?}", e),
        })?;
    }
    Ok(())
}

pub fn launch_nf4_gemm(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    nf4_weight: &Tensor<CudaRuntime>,
    absmax: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    m: u32,
    k: u32,
    n: u32,
    blocksize: u32,
) -> Result<()> {
    let device_index = input.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, NF4_QUANT_MODULE)?;
    let func = kernels::get_kernel_function(&module, "nf4_gemm_f32")?;

    let cfg = LaunchConfig {
        grid_dim: ((n + 15) / 16, (m + 15) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let input_ptr = input.ptr();
    let nf4_ptr = nf4_weight.ptr();
    let absmax_ptr = absmax.ptr();
    let output_ptr = output.ptr();

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&nf4_ptr);
        builder.arg(&absmax_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&blocksize);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA nf4_gemm launch failed: {:?}", e),
        })?;
    }
    Ok(())
}
