//! Paged decode attention — S_q=1 specialized fast path.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, PAGED_DECODE_ATTENTION_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Paged decode attention — S_q=1 specialized fast path
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
    let kernel_name = format!("paged_decode_attention_{}_fp32", head_dim);
    let device = q.device();
    let device_index = device.id();

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        PAGED_DECODE_ATTENTION_MODULE,
    )?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1, head_dim], DType::F32, device);
    // LSE not computed by decode kernel (not needed for inference)
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1], DType::F32, device);

    let max_num_blocks = block_table.shape()[1];
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
    let nh_i32 = num_heads as i32;
    let nkvh_i32 = num_kv_heads as i32;
    let sk_i32 = seq_len_k as i32;
    let mnb_i32 = max_num_blocks as i32;
    let bs_i32 = block_size as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&kb_ptr);
        builder.arg(&vb_ptr);
        builder.arg(&bt_ptr);
        builder.arg(&o_ptr);
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
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attention_fwd_graph(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_k_ptr: u64,
    head_dim: usize,
    block_size: usize,
    max_num_blocks: usize,
) -> Result<()> {
    let kernel_name = format!("paged_decode_attention_{}_fp32_graph", head_dim);
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
