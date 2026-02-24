//! MQA/GQA dedicated attention CUDA launchers
//!
//! Optimized kernels for extreme GQA ratios (num_kv_heads=1 for MQA).
//! For moderate ratios, the standard flash_v2 with num_kv_heads is used instead.
//!
//! Kernels: mqa_gqa.cu (fwd) + mqa_gqa_bwd.cu (bwd)

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash::set_smem_attribute;
use super::kernels::{self, MQA_GQA_BWD_MODULE, MQA_GQA_MODULE};

/// Block configuration for MQA/GQA kernels (head dims 32/64/128 only).
fn mqa_block_config(head_dim: usize) -> Result<(usize, usize)> {
    match head_dim {
        32 => Ok((128, 128)),
        64 => Ok((128, 128)),
        128 => Ok((128, 64)),
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "MQA/GQA kernels support head_dim 32/64/128, got {}",
                head_dim
            ),
        }),
    }
}

/// MQA/GQA forward pass — dedicated kernel for extreme GQA ratios.
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

    if num_heads % num_kv_heads != 0 {
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
    let (block_m, block_n) = mqa_block_config(head_dim)?;

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
    let kernel_name = format!("mqa_gqa_fwd_{}_{}", head_dim, dtype_suffix);

    let device = q.device();
    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device);
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device);

    let head_stride = head_dim + 1;
    let dtype_size = dtype.size_in_bytes();
    let smem_size = (block_m * head_stride + 2 * block_n * head_stride) * dtype_size;

    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, MQA_GQA_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (batch_size * num_heads) as u32,
            ((seq_len_q + block_m - 1) / block_m) as u32,
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

/// MQA/GQA backward pass.
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

    if num_heads % num_kv_heads != 0 {
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
    let (block_m, block_n) = mqa_block_config(head_dim)?;

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
        Tensor::<CudaRuntime>::zeros(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device);
    let dk = Tensor::<CudaRuntime>::zeros(
        &[batch_size, num_kv_heads, seq_len_k, head_dim],
        dtype,
        device,
    );
    let dv = Tensor::<CudaRuntime>::zeros(
        &[batch_size, num_kv_heads, seq_len_k, head_dim],
        dtype,
        device,
    );

    // Step 1: Preprocessing — D = rowsum(dO ⊙ O)
    let d_buf =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device);

    let module = kernels::get_or_load_module(client.context(), device_index, MQA_GQA_BWD_MODULE)?;

    {
        let preprocess_name = format!("mqa_gqa_preprocess_bwd_{}_{}", head_dim, dtype_suffix);
        let func = kernels::get_kernel_function(&module, &preprocess_name)?;

        let block_size = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                (seq_len_q as u32 + block_size - 1) / block_size,
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

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&dout_ptr);
            builder.arg(&out_ptr);
            builder.arg(&d_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("MQA/GQA bwd preprocess failed: {:?}", e),
            })?;
        }
    }

    // Step 2: Main backward — dQ, dK, dV
    {
        let bwd_name = format!("mqa_gqa_bwd_{}_{}", head_dim, dtype_suffix);
        let func = kernels::get_kernel_function(&module, &bwd_name)?;

        // Shared memory always F32 in shared memory (kernels convert internally)
        let smem_size = (2 * block_n * head_dim + 2 * block_m * head_dim) * 4;
        set_smem_attribute(&func, smem_size)?;

        let num_k_blocks = (seq_len_k + block_n - 1) / block_n;
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
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("MQA/GQA bwd kernel launch failed: {:?}", e),
            })?;
        }
    }

    Ok((dq, dk, dv))
}

/// Returns true if the GQA ratio warrants using dedicated MQA/GQA kernels.
///
/// Heuristic: use dedicated kernels when:
/// - num_heads / num_kv_heads >= 4 (extreme GQA ratio)
/// - head_dim is 32, 64, or 128 (supported by MQA/GQA kernels)
pub fn should_use_mqa_gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> bool {
    let ratio = num_heads / num_kv_heads;
    ratio >= 4 && matches!(head_dim, 32 | 64 | 128)
}
