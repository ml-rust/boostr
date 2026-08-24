//! MQA/GQA dedicated attention CUDA forward launcher
//!
//! Optimized kernel for extreme GQA ratios (num_kv_heads=1 for MQA).
//! For moderate ratios, the standard flash_v2 with num_kv_heads is used instead.
//!
//! Kernel: mqa_gqa.cu

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::flash_utils::{compute_smem, set_smem_attribute};
use super::block_config::mqa_fwd_block_config;
use crate::ops::cuda::kernels::{self, MQA_GQA_MODULE};

/// MQA/GQA forward pass — dedicated kernel for extreme GQA ratios.
#[allow(clippy::too_many_arguments)]
pub fn mqa_gqa_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let dtype = q.dtype();

    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            ),
        });
    }

    let batch_size = q_shape[0];
    let seq_len_q = q_shape[2];
    let seq_len_k = k_shape[2];

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {:?} for MQA/GQA", dtype),
            });
        }
    };

    // F32/F16/BF16 stage the tile in the tensor dtype, so the element size is
    // the dtype size. (The FP8 impls stage in f32 instead, but this launcher
    // rejects FP8 above.)
    let elem_bytes = dtype.size_in_bytes();
    let (block_m, block_n, use_sm_kernel) = mqa_fwd_block_config(head_dim, elem_bytes)?;

    let variant = if use_sm_kernel { "_sm" } else { "" };
    let kernel_name = format!("mqa_gqa_fwd_{}_{}{}", head_dim, dtype_suffix, variant);

    let device = q.device();
    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device)?;
    let lse =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    let smem_size = compute_smem(block_m, block_n, head_dim, elem_bytes);

    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, MQA_GQA_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;
    set_smem_attribute(&func, smem_size)?;

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
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let nkv_i32 = num_kv_heads as i32;
    let sq_i32 = seq_len_q as i32;
    let sk_i32 = seq_len_k as i32;
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
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("MQA/GQA fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
