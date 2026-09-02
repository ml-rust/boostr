//! Paged decode attention — S_q=1 specialized fast path.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, DECODE_ATTENTION_MODULE, PAGED_DECODE_ATTENTION_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::decode_split::{decode_dtype_suffix, decode_split_count};

/// Paged decode attention — S_q=1 specialized fast path.
///
/// The grid is one block per `(batch, Q head)` pair, which does not grow with
/// `seq_len_k`. When that leaves the device underfilled, the KV blocks are cut
/// into slices and the shared combine pass merges their partial softmax
/// statistics — see [`decode_split_count`].
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_decode_attention_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let suffix = decode_dtype_suffix(q.dtype())?;
    let stem = format!("paged_decode_attention_{head_dim}_{suffix}");
    let device = q.device();
    let device_index = device.id();

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        PAGED_DECODE_ATTENTION_MODULE,
    )?;

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1, head_dim], q.dtype(), device)?;
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1], DType::F32, device)?;

    let max_num_blocks = block_table.shape()[1];
    let scale = (head_dim as f32).sqrt().recip();

    let base_blocks = batch_size * num_heads;
    // A slice boundary lands on a KV block boundary, so the sequence cannot be
    // cut into more slices than it has blocks.
    let num_kv_blocks = seq_len_k.div_ceil(block_size);
    let splits = decode_split_count(device_index, base_blocks, seq_len_k).min(num_kv_blocks.max(1));

    let q_ptr = q.ptr();
    let kb_ptr = k_blocks.ptr();
    let vb_ptr = v_blocks.ptr();
    let bt_ptr = block_table.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = num_heads as i32;
    let nkvh_i32 = num_kv_heads as i32;
    let sk_i32 = seq_len_k as i32;
    let mnb_i32 = max_num_blocks as i32;
    let bs_i32 = block_size as i32;

    if splits > 1 {
        // Unnormalized per-slice accumulators plus their (m, l) statistics, in
        // the layout the contiguous decode path's combine kernel reads.
        let partial_o =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, head_dim], DType::F32, device)?;
        let partial_ml =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, 2], DType::F32, device)?;
        let po_ptr = partial_o.ptr();
        let pml_ptr = partial_ml.ptr();
        let splits_i32 = splits as i32;

        let split_func = kernels::get_kernel_function(&module, &format!("{stem}_split"))?;
        let split_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, splits as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&split_func);
            builder.arg(&q_ptr);
            builder.arg(&kb_ptr);
            builder.arg(&vb_ptr);
            builder.arg(&bt_ptr);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&nh_i32);
            builder.arg(&nkvh_i32);
            builder.arg(&sk_i32);
            builder.arg(&mnb_i32);
            builder.arg(&bs_i32);
            builder.arg(&scale);
            builder.arg(&splits_i32);
            builder.launch(split_cfg).map_err(|e| Error::KernelError {
                reason: format!("Paged decode split kernel launch failed: {:?}", e),
            })?;
        }

        // The partials carry no paging structure, so the contiguous decode
        // path's combine kernel merges them unchanged.
        let combine_module =
            kernels::get_or_load_module(client.context(), device_index, DECODE_ATTENTION_MODULE)?;
        let combine_func = kernels::get_kernel_function(
            &combine_module,
            &format!("decode_attention_{head_dim}_{suffix}_combine"),
        )?;
        let combine_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
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
                    reason: format!("Paged decode combine kernel launch failed: {:?}", e),
                })?;
        }

        return Ok((output, lse));
    }

    let func = kernels::get_kernel_function(&module, &stem)?;
    let cfg = LaunchConfig {
        grid_dim: (base_blocks as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&kb_ptr);
        builder.arg(&vb_ptr);
        builder.arg(&bt_ptr);
        builder.arg(&o_ptr);
        builder.arg(&lse_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkvh_i32);
        builder.arg(&sk_i32);
        builder.arg(&mnb_i32);
        builder.arg(&bs_i32);
        builder.arg(&scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Paged decode attention kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}

/// Paged decode attention — graph-mode variant that reads seq_len_k from device memory.
///
/// `seq_len_k_ptr` is a device pointer to an i32. Updated before each graph replay
/// via `cuMemsetD32Async`.
///
/// `output` and `lse` are caller-owned so they outlive the capture; a tensor
/// allocated here would be freed while the captured graph still writes to it.
/// The grid stays one block per `(batch, Q head)`: the split path needs scratch
/// buffers with the same lifetime, which the capture does not yet provide.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attention_fwd_graph(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_k_ptr: u64,
    head_dim: usize,
    block_size: usize,
    max_num_blocks: usize,
) -> Result<()> {
    let kernel_name = format!(
        "paged_decode_attention_{head_dim}_{}_graph",
        decode_dtype_suffix(q.dtype())?
    );
    let device = q.device();
    let device_index = device.id();

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        PAGED_DECODE_ATTENTION_MODULE,
    )?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let scale = (head_dim as f32).sqrt().recip();

    let cfg = LaunchConfig {
        grid_dim: ((batch_size * num_heads) as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let q_ptr = q.ptr();
    let kb_ptr = k_blocks.ptr();
    let vb_ptr = v_blocks.ptr();
    let bt_ptr = block_table.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = num_heads as i32;
    let nkvh_i32 = num_kv_heads as i32;
    let mnb_i32 = max_num_blocks as i32;
    let bs_i32 = block_size as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&kb_ptr);
        builder.arg(&vb_ptr);
        builder.arg(&bt_ptr);
        builder.arg(&o_ptr);
        builder.arg(&lse_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkvh_i32);
        builder.arg(&seq_len_k_ptr); // device pointer to i32
        builder.arg(&mnb_i32);
        builder.arg(&bs_i32);
        builder.arg(&scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Paged decode attention graph kernel launch failed: {:?}", e),
        })?;
    }

    Ok(())
}
