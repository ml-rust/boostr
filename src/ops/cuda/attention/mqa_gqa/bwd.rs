//! MQA/GQA dedicated attention CUDA backward launcher
//!
//! Kernel: mqa_gqa_bwd.cu

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::flash_utils::{compute_bwd_smem, set_smem_attribute};
use super::block_config::{BWD_SMEM_ELEM_BYTES, mqa_bwd_block_config};
use crate::ops::cuda::kernels::{self, MQA_GQA_BWD_MODULE};

/// MQA/GQA backward pass.
#[allow(clippy::too_many_arguments)]
pub fn mqa_gqa_bwd(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let q_shape = q.shape();
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
    let seq_len_k = k.shape()[2];
    let (block_m, block_n, use_sm_kernel) = mqa_bwd_block_config(head_dim)?;

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {:?} for MQA/GQA bwd", dtype),
            });
        }
    };

    let device = q.device();
    let device_index = device.id();

    // Allocate gradients (dQ zeroed for atomicAdd)
    let dq =
        Tensor::<CudaRuntime>::zeros(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device)?;
    let dk = Tensor::<CudaRuntime>::zeros(
        &[batch_size, num_kv_heads, seq_len_k, head_dim],
        dtype,
        device,
    )?;
    let dv = Tensor::<CudaRuntime>::zeros(
        &[batch_size, num_kv_heads, seq_len_k, head_dim],
        dtype,
        device,
    )?;

    // Step 1: Preprocessing — D = rowsum(dO ⊙ O)
    let d_buf =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    let module = kernels::get_or_load_module(client.context(), device_index, MQA_GQA_BWD_MODULE)?;

    {
        let preprocess_name = format!("mqa_gqa_preprocess_bwd_{}_{}", head_dim, dtype_suffix);
        let func = kernels::get_kernel_function(&module, &preprocess_name)?;

        let block_size = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                (seq_len_q as u32).div_ceil(block_size),
                1,
            ),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let dout_ptr = dout.ptr();
        let out_ptr = output.ptr();
        let d_ptr = d_buf.ptr();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        // The F32/BF16 preprocess kernels declare trailing dequant scales and
        // ignore them; the F16 kernel declares neither. Push them either way so
        // no kernel reads a parameter slot that was never written.
        let scale_do = 1.0f32;
        let scale_o = 1.0f32;

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&dout_ptr);
            builder.arg(&out_ptr);
            builder.arg(&d_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&scale_do);
            builder.arg(&scale_o);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("MQA/GQA bwd preprocess failed: {:?}", e),
            })?;
        }
    }

    // Step 2: Main backward — dQ, dK, dV
    {
        let variant = if use_sm_kernel { "_sm" } else { "" };
        let bwd_name = format!("mqa_gqa_bwd_{}_{}{}", head_dim, dtype_suffix, variant);
        let func = kernels::get_kernel_function(&module, &bwd_name)?;

        // Shared memory is always F32 (kernels convert on load), never the tensor dtype.
        let smem_size = compute_bwd_smem(block_m, block_n, head_dim, BWD_SMEM_ELEM_BYTES);
        set_smem_attribute(&func, smem_size)?;

        let num_k_blocks = seq_len_k.div_ceil(block_n);
        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, num_k_blocks as u32, 1),
            block_dim: (block_m as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let o_ptr = output.ptr();
        let dout_ptr = dout.ptr();
        let lse_ptr = lse.ptr();
        let d_ptr = d_buf.ptr();
        let dq_ptr = dq.ptr();
        let dk_ptr = dk.ptr();
        let dv_ptr = dv.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let nkv_i32 = num_kv_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };
        // The F32/BF16 kernels declare eight trailing quantization scales and
        // ignore them; the F16 kernel declares none. Push them either way so no
        // kernel reads a parameter slot that was never written.
        let one = 1.0f32;

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&o_ptr);
            builder.arg(&dout_ptr);
            builder.arg(&lse_ptr);
            builder.arg(&d_ptr);
            builder.arg(&dq_ptr);
            builder.arg(&dk_ptr);
            builder.arg(&dv_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&nkv_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            // scale_q, scale_k, scale_v, scale_o, scale_do, scale_dq, scale_dk, scale_dv
            for _ in 0..8 {
                builder.arg(&one);
            }
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("MQA/GQA bwd kernel launch failed: {:?}", e),
            })?;
        }
    }

    Ok((dq, dk, dv))
}
