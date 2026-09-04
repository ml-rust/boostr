//! CUDA launchers for INT4 KV cache quantize/dequantize.
//!
//! Split out of `kv_cache_quant.rs` to keep that file under the `cuda/*.rs`
//! line limit. Kernels: `kv_cache_int4.cu`.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_INT4_MODULE};
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

pub(super) fn quantize_kv_int4_impl(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    num_tokens: usize,
    head_dim: usize,
    group_size: Int4GroupSize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let dtype = input.dtype();
    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::KernelError {
                reason: format!("INT4 quant: unsupported dtype {dtype:?}"),
            });
        }
    };

    let kernel_name = format!("quantize_kv_int4_per_group_{dtype_suffix}");
    let device = input.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_CACHE_INT4_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let gs = group_size as usize;
    let total = num_tokens * head_dim;
    let num_groups = total.div_ceil(gs);

    let packed = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim / 2], DType::U8, device)?;
    let scales_t = Tensor::<CudaRuntime>::empty(&[num_groups], DType::F32, device)?;
    let zeros_t = Tensor::<CudaRuntime>::empty(&[num_groups], DType::F32, device)?;

    let cfg = LaunchConfig {
        grid_dim: (num_groups as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    let i_ptr = input.ptr();
    let p_ptr = packed.ptr();
    let s_ptr = scales_t.ptr();
    let z_ptr = zeros_t.ptr();
    let nt_i32 = num_tokens as i32;
    let hd_i32 = head_dim as i32;
    let gs_i32 = gs as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&i_ptr);
        builder.arg(&p_ptr);
        builder.arg(&s_ptr);
        builder.arg(&z_ptr);
        builder.arg(&nt_i32);
        builder.arg(&hd_i32);
        builder.arg(&gs_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("INT4 quant failed: {e:?}"),
        })?;
    }

    Ok((packed, scales_t, zeros_t))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dequantize_kv_int4_impl(
    client: &CudaClient,
    packed: &Tensor<CudaRuntime>,
    scales: &Tensor<CudaRuntime>,
    zeros: &Tensor<CudaRuntime>,
    num_tokens: usize,
    head_dim: usize,
    group_size: Int4GroupSize,
    output_dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let kernel_name = match output_dtype {
        DType::F32 => "dequantize_kv_int4_per_group_fp32",
        DType::F16 => "dequantize_kv_int4_per_group_fp16",
        DType::BF16 => "dequantize_kv_int4_per_group_bf16",
        _ => {
            return Err(Error::KernelError {
                reason: format!("INT4 dequant: unsupported output dtype {output_dtype:?}"),
            });
        }
    };

    let device = packed.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_CACHE_INT4_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let gs = group_size as usize;
    let total = num_tokens * head_dim;
    let num_groups = total.div_ceil(gs);

    let output = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], output_dtype, device)?;

    let cfg = LaunchConfig {
        grid_dim: (num_groups as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let p_ptr = packed.ptr();
    let o_ptr = output.ptr();
    let s_ptr = scales.ptr();
    let z_ptr = zeros.ptr();
    let nt_i32 = num_tokens as i32;
    let hd_i32 = head_dim as i32;
    let gs_i32 = gs as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&p_ptr);
        builder.arg(&s_ptr);
        builder.arg(&z_ptr);
        builder.arg(&o_ptr);
        builder.arg(&nt_i32);
        builder.arg(&hd_i32);
        builder.arg(&gs_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("INT4 dequant failed: {e:?}"),
        })?;
    }

    Ok(output)
}
