//! Flash Attention v2 forward pass implementations (F32/F16/BF16 and FP8).

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FLASH_V2_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{AttentionParams, set_smem_attribute};

/// Forward pass for F32/F16/BF16 — main tiled Flash Attention v2 kernel.
pub(super) fn flash_attention_fwd_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    causal: bool,
    window_size: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let dtype = q.dtype();

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "unsupported dtype {:?} for flash_attention_fwd. Use flash_attention_fwd_fp8 for FP8.",
                    dtype
                ),
            });
        }
    };
    let sm_suffix = if p.use_sm_kernel { "_sm" } else { "" };
    let kernel_name = format!(
        "flash_attention_fwd_{}{}_{}",
        p.head_dim, sm_suffix, dtype_suffix
    );

    let device = q.device();
    let output = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
        dtype,
        device,
    );
    let lse = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q],
        DType::F32,
        device,
    );

    let head_stride = p.head_dim + 1;
    let dtype_size = dtype.size_in_bytes();
    let smem_size = (p.block_m * head_stride + 2 * p.block_n * head_stride) * dtype_size;

    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, FLASH_V2_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (p.batch_size * p.num_heads) as u32,
            p.seq_len_q.div_ceil(p.block_m) as u32,
            1,
        ),
        block_dim: (p.block_m as u32, 1, 1),
        shared_mem_bytes: smem_size as u32,
    };

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let scale = (p.head_dim as f32).sqrt().recip();
    let batch_i32 = p.batch_size as i32;
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sq_i32 = p.seq_len_q as i32;
    let sk_i32 = p.seq_len_k as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };
    let ws_i32 = window_size as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&sq_i32);
        builder.arg(&sk_i32);
        builder.arg(&scale);
        builder.arg(&causal_i32);
        builder.arg(&ws_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Flash Attention fwd kernel launch failed: {:?}", e),
        })?;
    }

    // No sync needed: same-stream ordering guarantees the kernel
    // completes before any subsequent kernel on this stream.

    Ok((output, lse))
}

/// Forward pass for FP8 (E4M3/E5M2) — separate scale args per tensor.
#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_fwd_fp8_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    causal: bool,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    o_scale: f32,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let dtype = q.dtype();

    // FP8 kernels use E4M3 format (kernel handles both via same entry point)
    let sm_suffix = if p.use_sm_kernel { "_sm" } else { "" };
    let kernel_name = format!("flash_attention_fwd_{}{}_fp8", p.head_dim, sm_suffix);

    let device = q.device();
    let output = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
        dtype,
        device,
    );
    let lse = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q],
        DType::F32,
        device,
    );

    // FP8 is 1 byte per element
    let head_stride = p.head_dim + 1;
    let smem_size = p.block_m * head_stride + 2 * p.block_n * head_stride;

    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, FLASH_V2_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (p.batch_size * p.num_heads) as u32,
            p.seq_len_q.div_ceil(p.block_m) as u32,
            1,
        ),
        block_dim: (p.block_m as u32, 1, 1),
        shared_mem_bytes: smem_size as u32,
    };

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let scale = (p.head_dim as f32).sqrt().recip();
    let batch_i32 = p.batch_size as i32;
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sq_i32 = p.seq_len_q as i32;
    let sk_i32 = p.seq_len_k as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&sq_i32);
        builder.arg(&sk_i32);
        builder.arg(&scale);
        builder.arg(&causal_i32);
        builder.arg(&q_scale);
        builder.arg(&k_scale);
        builder.arg(&v_scale);
        builder.arg(&o_scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Flash Attention FP8 fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
