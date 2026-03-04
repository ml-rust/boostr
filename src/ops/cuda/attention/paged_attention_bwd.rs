//! Paged attention backward kernel launcher.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, PAGED_ATTENTION_BWD_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::paged_attention::bwd_block_config;

/// Paged attention backward kernel launcher.
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_attention_bwd_impl(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
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
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {:?} for paged attention backward", dtype),
            });
        }
    };

    let (block_m, block_n) = bwd_block_config(head_dim, dtype)?;
    let kernel_name = format!(
        "paged_flash_attention_bwd_{}_{}_small",
        head_dim, dtype_suffix
    );

    let device = q.device();
    let max_num_blocks = block_table.shape()[1];

    // dQ: contiguous [B, H, S_q, D]
    let dq =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device);
    // dK, dV: same shape as k_blocks/v_blocks — zeroed for atomicAdd
    let dk_blocks = Tensor::<CudaRuntime>::zeros(k_blocks.shape(), dtype, device);
    let dv_blocks = Tensor::<CudaRuntime>::zeros(v_blocks.shape(), dtype, device);

    let dtype_size = dtype.size_in_bytes();
    let smem_size = (3 * block_m * head_dim + 2 * block_n * head_dim) * dtype_size;

    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, PAGED_ATTENTION_BWD_MODULE)?;
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
    let o_ptr = output.ptr();
    let do_ptr = dout.ptr();
    let l_ptr = lse.ptr();
    let bt_ptr = block_table.ptr();
    let dq_ptr = dq.ptr();
    let dkb_ptr = dk_blocks.ptr();
    let dvb_ptr = dv_blocks.ptr();
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
        builder.arg(&o_ptr);
        builder.arg(&do_ptr);
        builder.arg(&l_ptr);
        builder.arg(&bt_ptr);
        builder.arg(&dq_ptr);
        builder.arg(&dkb_ptr);
        builder.arg(&dvb_ptr);
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
            reason: format!("Paged attention bwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((dq, dk_blocks, dv_blocks))
}
