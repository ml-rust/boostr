//! CUDA launchers for INT8 KV cache quantize/dequantize.
//!
//! Split out of `kv_cache_quant.rs` to keep that file under the `cuda/*.rs`
//! line limit. Kernels: `kv_cache_quant.cu`.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_QUANT_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

pub(super) fn quantize_kv_int8_impl(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    num_tokens: usize,
    head_dim: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let dtype = input.dtype();
    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::KernelError {
                reason: format!("INT8 quant: unsupported dtype {dtype:?}"),
            });
        }
    };

    let kernel_name = format!("quantize_kv_int8_per_token_{dtype_suffix}");
    let device = input.device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, KV_CACHE_QUANT_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let quantized = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::I8, device)?;
    let scales = Tensor::<CudaRuntime>::empty(&[num_tokens], DType::F32, device)?;

    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    let i_ptr = input.ptr();
    let q_ptr = quantized.ptr();
    let s_ptr = scales.ptr();
    let nt_i32 = num_tokens as i32;
    let hd_i32 = head_dim as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&i_ptr);
        builder.arg(&q_ptr);
        builder.arg(&s_ptr);
        builder.arg(&nt_i32);
        builder.arg(&hd_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("INT8 quant failed: {e:?}"),
        })?;
    }

    Ok((quantized, scales))
}

pub(super) fn dequantize_kv_int8_impl(
    client: &CudaClient,
    quantized: &Tensor<CudaRuntime>,
    scales: &Tensor<CudaRuntime>,
    num_tokens: usize,
    head_dim: usize,
) -> Result<Tensor<CudaRuntime>> {
    let device = quantized.device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, KV_CACHE_QUANT_MODULE)?;
    let func = kernels::get_kernel_function(&module, "dequantize_kv_int8_per_token_fp32")?;

    let output = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::F32, device)?;

    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let q_ptr = quantized.ptr();
    let o_ptr = output.ptr();
    let s_ptr = scales.ptr();
    let nt_i32 = num_tokens as i32;
    let hd_i32 = head_dim as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&o_ptr);
        builder.arg(&s_ptr);
        builder.arg(&nt_i32);
        builder.arg(&hd_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("INT8 dequant failed: {e:?}"),
        })?;
    }

    Ok(output)
}
