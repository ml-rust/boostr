//! Flash Attention decode path: lightweight vec kernels for S_q=1.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::decode_split::decode_split_count;
use super::flash_utils::AttentionParams;

/// Kernel name stem for a supported decode `head_dim`.
fn decode_kernel_stem(head_dim: usize) -> Result<&'static str> {
    match head_dim {
        64 => Ok("decode_attention_64_fp32"),
        128 => Ok("decode_attention_128_fp32"),
        other => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!("decode attention supports head_dim 64/128, got {other}"),
        }),
    }
}

/// Decode attention for S_q=1: lightweight vec kernel, no tiling.
///
/// The grid is one block per `(batch, head)` pair, which does not grow with
/// `seq_len_k`. When that leaves the device underfilled, the KV sequence is cut
/// into slices and a combine pass merges their partial softmax statistics —
/// see [`decode_split_count`].
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
    let stem = decode_kernel_stem(p.head_dim)?;

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;

    let output = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, 1, p.head_dim],
        q.dtype(),
        device,
    )?;
    let lse = Tensor::<CudaRuntime>::empty(&[p.batch_size, p.num_heads, 1], DType::F32, device)?;

    let base_blocks = p.batch_size * p.num_heads;
    let splits = decode_split_count(device_index, base_blocks, p.seq_len_k);

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sk_i32 = p.seq_len_k as i32;
    let stride_i32 = kv_seq_stride as i32;
    let scale = (p.head_dim as f32).sqrt().recip();

    if splits > 1 {
        // Unnormalized per-slice accumulators plus their (m, l) statistics.
        let partial_o =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, p.head_dim], DType::F32, device)?;
        let partial_ml =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, 2], DType::F32, device)?;
        let po_ptr = partial_o.ptr();
        let pml_ptr = partial_ml.ptr();
        let splits_i32 = splits as i32;

        let split_func = kernels::get_kernel_function(&module, &format!("{stem}_split"))?;
        let split_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, splits as u32, 1),
            block_dim: (p.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&split_func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&nh_i32);
            builder.arg(&nkv_i32);
            builder.arg(&sk_i32);
            builder.arg(&stride_i32);
            builder.arg(&scale);
            builder.arg(&splits_i32);
            builder.launch(split_cfg).map_err(|e| Error::KernelError {
                reason: format!("decode_attention split kernel launch failed: {:?}", e),
            })?;
        }

        let combine_func = kernels::get_kernel_function(&module, &format!("{stem}_combine"))?;
        let combine_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, 1, 1),
            block_dim: (p.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&combine_func);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&o_ptr);
            builder.arg(&lse_ptr);
            builder.arg(&splits_i32);
            builder
                .launch(combine_cfg)
                .map_err(|e| Error::KernelError {
                    reason: format!("decode_attention combine kernel launch failed: {:?}", e),
                })?;
        }

        return Ok((output, lse));
    }

    let func = kernels::get_kernel_function(&module, stem)?;
    let cfg = LaunchConfig {
        grid_dim: (base_blocks as u32, 1, 1),
        block_dim: (p.head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&lse_ptr);
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
///
/// `window_size` is the sliding-window span; `0` disables it, matching every
/// other call path. Decode is single-token, so the query sits at absolute
/// position `seq_len_k - 1` and the kernel keeps keys `j >= seq_len_k -
/// window_size`. It is a static config value, not a per-step one, so passing it
/// as a plain scalar is safe under CUDA graph capture — unlike `seq_len_k`,
/// which changes every replay and therefore stays a device pointer.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
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
    window_size: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let device = q.device();
    let device_index = device.id();
    let batch_size = q.shape()[0];

    // Unlike the non-graph path, nothing upstream of graph mode filters head_dim,
    // so an unsupported one is an error, not an unreachable case.
    let kernel_name = format!("{}_graph", decode_kernel_stem(head_dim)?);

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1, head_dim], q.dtype(), device)?;
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1], DType::F32, device)?;

    let q_ptr = q.ptr();
    let k_ptr = k_cache.ptr();
    let v_ptr = v_cache.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = num_heads as i32;
    let nkv_i32 = num_kv_heads as i32;
    let stride_i32 = kv_capacity as i32;
    let window_i32 = window_size as i32;
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
        builder.arg(&lse_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&seq_len_k_ptr);
        builder.arg(&stride_i32);
        builder.arg(&scale);
        builder.arg(&window_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("decode_attention_graph kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
