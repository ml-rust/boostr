//! Paged attention forward kernel launchers (standard and FP8).

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, PAGED_ATTENTION_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::paged_attention::{bwd_block_config, fwd_block_config};
use super::paged_decode::paged_decode_attention_fwd;

/// Standard (non-FP8) paged attention forward.
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_attention_fwd_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    let batch_size = q_shape[0];
    let dtype = q.dtype();

    // Fast path: S_q=1 decode with specialized kernel (no shared memory tiling overhead)
    if seq_len_q == 1 && dtype == DType::F32 && (head_dim == 64 || head_dim == 128) {
        return paged_decode_attention_fwd(
            client,
            q,
            k_blocks,
            v_blocks,
            block_table,
            batch_size,
            num_heads,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
        );
    }

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "unsupported dtype {:?}. Use paged_attention_fwd_fp8 for FP8.",
                    dtype
                ),
            });
        }
    };

    let (block_m, block_n) = fwd_block_config(head_dim, dtype)?;
    let kernel_name = format!(
        "paged_flash_attention_fwd_{}_{}_small",
        head_dim, dtype_suffix
    );

    let device = q.device();
    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device);
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device);

    let dtype_size = dtype.size_in_bytes();
    let smem_size = (block_m * head_dim + block_n * head_dim + block_n * head_dim) * dtype_size;

    let max_num_blocks = block_table.shape()[1];
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, PAGED_ATTENTION_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (batch_size * num_heads) as u32,
            seq_len_q.div_ceil(block_m) as u32,
            1,
        ),
        block_dim: (block_m as u32, 1, 1),
        shared_mem_bytes: smem_size as u32,
    };

    let q_ptr = q.ptr();
    let kb_ptr = k_blocks.ptr();
    let vb_ptr = v_blocks.ptr();
    let bt_ptr = block_table.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let nkvh_i32 = num_kv_heads as i32;
    let sq_i32 = seq_len_q as i32;
    let sk_i32 = seq_len_k as i32;
    let mnb_i32 = max_num_blocks as i32;
    let bs_i32 = block_size as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&kb_ptr);
        builder.arg(&vb_ptr);
        builder.arg(&bt_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&nkvh_i32);
        builder.arg(&sq_i32);
        builder.arg(&sk_i32);
        builder.arg(&mnb_i32);
        builder.arg(&bs_i32);
        builder.arg(&scale);
        builder.arg(&causal_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Paged attention fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}

/// FP8 paged attention forward.
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_attention_fwd_fp8_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    o_scale: f32,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    let batch_size = q_shape[0];
    let dtype = q.dtype();

    let dtype_suffix = match dtype {
        DType::FP8E4M3 => "fp8_e4m3",
        DType::FP8E5M2 => "fp8_e5m2",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "paged_attention_fwd_fp8 requires FP8 dtype, got {:?}",
                    dtype
                ),
            });
        }
    };

    let (block_m, _block_n) = fwd_block_config(head_dim, dtype)?;
    let kernel_name = format!(
        "paged_flash_attention_fwd_{}_{}_small",
        head_dim, dtype_suffix
    );

    let device = q.device();
    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device);
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device);

    // FP8 uses FP32 smem: (BLOCK_M + 2*BLOCK_N) * HEAD_DIM * 4
    let (bm, bn) = fwd_block_config(head_dim, dtype)?;
    let smem_size = (bm * head_dim + 2 * bn * head_dim) * 4;

    let max_num_blocks = block_table.shape()[1];
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, PAGED_ATTENTION_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (batch_size * num_heads) as u32,
            seq_len_q.div_ceil(block_m) as u32,
            1,
        ),
        block_dim: (block_m as u32, 1, 1),
        shared_mem_bytes: smem_size as u32,
    };

    let q_ptr = q.ptr();
    let kb_ptr = k_blocks.ptr();
    let vb_ptr = v_blocks.ptr();
    let bt_ptr = block_table.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let attn_scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let nkvh_i32 = num_kv_heads as i32;
    let sq_i32 = seq_len_q as i32;
    let sk_i32 = seq_len_k as i32;
    let mnb_i32 = max_num_blocks as i32;
    let bs_i32 = block_size as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&kb_ptr);
        builder.arg(&vb_ptr);
        builder.arg(&bt_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&nkvh_i32);
        builder.arg(&sq_i32);
        builder.arg(&sk_i32);
        builder.arg(&mnb_i32);
        builder.arg(&bs_i32);
        builder.arg(&attn_scale);
        builder.arg(&q_scale);
        builder.arg(&k_scale);
        builder.arg(&v_scale);
        builder.arg(&o_scale);
        builder.arg(&causal_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Paged attention FP8 fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
