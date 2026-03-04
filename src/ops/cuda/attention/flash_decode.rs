//! Flash Attention decode path: lightweight vec kernels for S_q=1.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::AttentionParams;

/// Decode attention for S_q=1: lightweight vec kernel, no tiling.
/// One block per (batch, head), head_dim threads. Online softmax in registers.
///
/// Non-graph path: seq_len_k passed as plain i32 kernel arg (zero overhead).
pub(super) fn decode_attention_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let device = q.device();
    let device_index = device.id();

    let kernel_name = match p.head_dim {
        64 => "decode_attention_64_fp32",
        128 => "decode_attention_128_fp32",
        _ => unreachable!("decode attention only supports head_dim 64/128"),
    };

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let output = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, 1, p.head_dim],
        q.dtype(),
        device,
    );
    let lse = Tensor::<CudaRuntime>::empty(&[p.batch_size, p.num_heads, 1], DType::F32, device);

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sk_i32 = p.seq_len_k as i32;
    let stride_i32 = kv_seq_stride as i32;
    let scale = (p.head_dim as f32).sqrt().recip();

    let num_blocks = p.batch_size * p.num_heads;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (p.head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&sk_i32);
        builder.arg(&stride_i32);
        builder.arg(&scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("decode_attention kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}

/// Graph-mode decode attention: uses `_graph` kernel variants with device-pointer
/// seq_len_k and separate kv_seq_stride for full-capacity raw KV buffers.
#[cfg(feature = "cuda")]
pub fn decode_attention_graph_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_cache: &Tensor<CudaRuntime>,
    v_cache: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len_k_ptr: u64,
    kv_capacity: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let device = q.device();
    let device_index = device.id();
    let batch_size = q.shape()[0];

    let kernel_name = match head_dim {
        64 => "decode_attention_64_fp32_graph",
        128 => "decode_attention_128_fp32_graph",
        _ => unreachable!("decode attention only supports head_dim 64/128"),
    };

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1, head_dim], q.dtype(), device);
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1], DType::F32, device);

    let q_ptr = q.ptr();
    let k_ptr = k_cache.ptr();
    let v_ptr = v_cache.ptr();
    let o_ptr = output.ptr();
    let nh_i32 = num_heads as i32;
    let nkv_i32 = num_kv_heads as i32;
    let stride_i32 = kv_capacity as i32;
    let scale = (head_dim as f32).sqrt().recip();

    let num_blocks = batch_size * num_heads;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&seq_len_k_ptr);
        builder.arg(&stride_i32);
        builder.arg(&scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("decode_attention_graph kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
