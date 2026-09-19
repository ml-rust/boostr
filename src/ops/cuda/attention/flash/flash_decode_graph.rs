//! Graph-capturable decode attention for a single query row.
//!
//! Same kernels as `flash_decode.rs`, but `seq_len_k` is a device pointer
//! and the grid is sized from the cache capacity, so the launch can be
//! captured into a CUDA graph and replayed as the cache grows.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::decode_split::{decode_kv_span, decode_slices};
use super::flash_decode::decode_kernel_stem;

/// Graph-mode decode attention: uses `_graph` kernel variants with device-pointer
/// seq_len_k and separate kv_seq_stride for full-capacity raw KV buffers.
///
/// `window_size` is the sliding-window span; `0` disables it, matching every
/// other call path. Decode is single-token, so the query sits at absolute
/// position `seq_len_k - 1` and the kernel keeps keys `j >= seq_len_k -
/// window_size`. It is a static config value, not a per-step one, so passing it
/// as a plain scalar is safe under CUDA graph capture — unlike `seq_len_k`,
/// which changes every replay and therefore stays a device pointer.
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
    let stem = decode_kernel_stem(head_dim, q.dtype())?;

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;

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

    let base_blocks = batch_size * num_heads;

    // The grid is baked in at capture time, so the split count cannot come from
    // the device-resident seq_len_k (only known per-replay, not at capture) — it
    // comes from kv_capacity, the static upper bound. At replay, slices whose
    // `[begin, end)` falls past the real seq_len_k are empty; the split kernel's
    // `begin < end` guard and the combine kernel's `l <= 0` guard both skip them
    // for free. Consequence: at early decode steps, with the cache nearly empty,
    // most slices do no work — correct, but the grid stays sized for a full cache
    // every step, not just the steps that need it. A window caps the span the
    // kernel walks at any step, so it caps the grid too.
    let slices = decode_slices(
        device_index,
        num_heads,
        decode_kv_span(kv_capacity, window_size),
        head_dim,
    );
    let splits = slices.splits;

    if splits > 1 {
        // Unnormalized per-slice accumulators plus their (m, l) statistics.
        // Allocated inside the capture closure: numr's allocator is frozen during
        // capture, so this becomes a graph alloc/free node pair replayed every
        // launch, exactly like the non-graph split path's scratch.
        let partial_o =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, head_dim], DType::F32, device)?;
        let partial_ml =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, 2], DType::F32, device)?;
        let po_ptr = partial_o.ptr();
        let pml_ptr = partial_ml.ptr();
        let splits_i32 = splits as i32;
        // The grid is sized for the capacity; at replay each row cuts its live
        // span by the same rule, and the slices past it are empty.
        let fill_i32 = slices.fill as i32;

        let split_func = kernels::get_kernel_function(&module, &format!("{stem}_split_graph"))?;
        let split_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, splits as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
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
            builder.arg(&seq_len_k_ptr);
            builder.arg(&stride_i32);
            builder.arg(&scale);
            builder.arg(&window_i32);
            builder.arg(&fill_i32);
            builder.launch(split_cfg).map_err(|e| Error::KernelError {
                reason: format!("decode_attention_graph split kernel launch failed: {:?}", e),
            })?;
        }

        // Static num_splits, so the combine kernel is capture-safe unchanged —
        // the same entry point the non-graph split path already uses.
        let combine_func = kernels::get_kernel_function(&module, &format!("{stem}_combine"))?;
        let combine_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // One query row: head-major and token-major are the same bytes, so
        // the combine stores row `bh` at `bh * D` (fold 1).
        let fold_i32 = 1i32;
        unsafe {
            let mut builder = client.stream().launch_builder(&combine_func);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&o_ptr);
            builder.arg(&lse_ptr);
            builder.arg(&splits_i32);
            builder.arg(&nh_i32);
            builder.arg(&fold_i32);
            builder
                .launch(combine_cfg)
                .map_err(|e| Error::KernelError {
                    reason: format!(
                        "decode_attention_graph combine kernel launch failed: {:?}",
                        e
                    ),
                })?;
        }

        return Ok((output, lse));
    }

    let func = kernels::get_kernel_function(&module, &format!("{stem}_graph"))?;
    let cfg = LaunchConfig {
        grid_dim: (base_blocks as u32, 1, 1),
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
