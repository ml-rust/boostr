//! CUDA launcher for `kv_cache_fp8_bwd.cu`: backward of FP8 KV-cache
//! fake-quantization (straight-through estimator + scale gradient).
//!
//! Kernels: `kv_cache_fp8_bwd_per_{tensor,token}_{fp16,bf16,fp32}`.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_FP8_BWD_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Threads per block for every launch in this module.
///
/// MUST be a power of two and at most 256: the kernels' reduction tree halves
/// `blockDim.x` each step into a statically sized `__shared__ float
/// smem_sum[256]`, so a larger or non-power-of-two block reads/writes out of
/// that array's bounds.
const FP8_BWD_BLOCK: u32 = 256;

fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F16 => Ok("fp16"),
        DType::BF16 => Ok("bf16"),
        DType::F32 => Ok("fp32"),
        _ => Err(Error::InvalidArgument {
            arg: "grad_output",
            reason: format!("kv_fp8_bwd: unsupported dtype {dtype:?}, need F32/F16/BF16"),
        }),
    }
}

fn check_kv_fp8(t: &Tensor<CudaRuntime>) -> Result<()> {
    if t.dtype() != DType::FP8E4M3 {
        return Err(Error::InvalidArgument {
            arg: "kv_fp8",
            reason: format!("kv_fp8_bwd: kv_fp8 must be FP8E4M3, got {:?}", t.dtype()),
        });
    }
    Ok(())
}

fn check_scales_f32(t: &Tensor<CudaRuntime>, arg: &'static str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("kv_fp8_bwd: {arg} must be F32, got {:?}", t.dtype()),
        });
    }
    Ok(())
}

/// Backward for FP8 fake-quantization with a single tensor-wide scale.
/// Launches `kv_cache_fp8_bwd_per_tensor_{fp16,bf16,fp32}`.
pub(super) fn kv_fp8_bwd_per_tensor_impl(
    client: &CudaClient,
    grad_output: &Tensor<CudaRuntime>,
    kv_fp8: &Tensor<CudaRuntime>,
    scale: f32,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    check_kv_fp8(kv_fp8)?;
    let total_elements = grad_output.numel();
    if kv_fp8.numel() != total_elements {
        return Err(Error::InvalidArgument {
            arg: "kv_fp8",
            reason: format!(
                "kv_fp8_bwd_per_tensor: kv_fp8 has {} elements, grad_output has {}",
                kv_fp8.numel(),
                total_elements
            ),
        });
    }

    let dtype = grad_output.dtype();
    let suffix = dtype_suffix(dtype)?;
    let kernel_name = format!("kv_cache_fp8_bwd_per_tensor_{suffix}");

    let device = grad_output.device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, KV_CACHE_FP8_BWD_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let grad_kv = Tensor::<CudaRuntime>::empty(grad_output.shape(), dtype, device)?;
    // atomicAdd accumulates into this single scalar across every block, so it
    // must start at zero.
    let grad_scale = Tensor::<CudaRuntime>::zeros(&[1], DType::F32, device)?;

    let cfg = LaunchConfig {
        grid_dim: ((total_elements as u32).div_ceil(FP8_BWD_BLOCK), 1, 1),
        block_dim: (FP8_BWD_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let gk_ptr = grad_kv.ptr();
    let go_ptr = grad_output.ptr();
    let kv_ptr = kv_fp8.ptr();
    let gs_ptr = grad_scale.ptr();
    let total_i32 = total_elements as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&gk_ptr);
        builder.arg(&go_ptr);
        builder.arg(&kv_ptr);
        builder.arg(&scale);
        builder.arg(&total_i32);
        builder.arg(&gs_ptr);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("kv_fp8_bwd_per_tensor kernel failed: {e:?}"),
        })?;
    }

    Ok((grad_kv, grad_scale))
}

/// Backward for FP8 fake-quantization with one scale per token.
/// Launches `kv_cache_fp8_bwd_per_token_{fp16,bf16,fp32}`, one block per
/// token, `total_tokens = batch * num_kv_heads * seq_len`.
#[allow(clippy::too_many_arguments)]
pub(super) fn kv_fp8_bwd_per_token_impl(
    client: &CudaClient,
    grad_output: &Tensor<CudaRuntime>,
    kv_fp8: &Tensor<CudaRuntime>,
    scales: &Tensor<CudaRuntime>,
    batch: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    check_kv_fp8(kv_fp8)?;
    check_scales_f32(scales, "scales")?;

    let total_tokens = batch * num_kv_heads * seq_len;
    let expected_numel = total_tokens * head_dim;
    if grad_output.numel() != expected_numel {
        return Err(Error::InvalidArgument {
            arg: "grad_output",
            reason: format!(
                "kv_fp8_bwd_per_token: expected {expected_numel} elements (batch*num_kv_heads*seq_len*head_dim), got {}",
                grad_output.numel()
            ),
        });
    }
    if kv_fp8.numel() != expected_numel {
        return Err(Error::InvalidArgument {
            arg: "kv_fp8",
            reason: format!(
                "kv_fp8_bwd_per_token: kv_fp8 has {} elements, expected {expected_numel}",
                kv_fp8.numel()
            ),
        });
    }
    if scales.numel() != total_tokens {
        return Err(Error::InvalidArgument {
            arg: "scales",
            reason: format!(
                "kv_fp8_bwd_per_token: scales has {} elements, expected {total_tokens} (batch*num_kv_heads*seq_len)",
                scales.numel()
            ),
        });
    }

    let dtype = grad_output.dtype();
    let suffix = dtype_suffix(dtype)?;
    let kernel_name = format!("kv_cache_fp8_bwd_per_token_{suffix}");

    let device = grad_output.device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, KV_CACHE_FP8_BWD_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let grad_kv = Tensor::<CudaRuntime>::empty(grad_output.shape(), dtype, device)?;
    // One block per token writes its own `grad_scales[token]` slot directly
    // (no atomics), so no zero-init is needed here.
    let grad_scales = Tensor::<CudaRuntime>::empty(&[total_tokens], DType::F32, device)?;

    let cfg = LaunchConfig {
        grid_dim: (total_tokens as u32, 1, 1),
        block_dim: (FP8_BWD_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let gk_ptr = grad_kv.ptr();
    let go_ptr = grad_output.ptr();
    let kv_ptr = kv_fp8.ptr();
    let s_ptr = scales.ptr();
    let gs_ptr = grad_scales.ptr();
    let batch_i32 = batch as i32;
    let nkh_i32 = num_kv_heads as i32;
    let sl_i32 = seq_len as i32;
    let hd_i32 = head_dim as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&gk_ptr);
        builder.arg(&go_ptr);
        builder.arg(&kv_ptr);
        builder.arg(&s_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nkh_i32);
        builder.arg(&sl_i32);
        builder.arg(&hd_i32);
        builder.arg(&gs_ptr);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("kv_fp8_bwd_per_token kernel failed: {e:?}"),
        })?;
    }

    Ok((grad_kv, grad_scales))
}
