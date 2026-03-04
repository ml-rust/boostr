//! Flash Attention v2 backward pass (F32/F16/BF16): dQ, dK, dV computation.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FLASH_V2_BWD_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{AttentionParams, set_smem_attribute};

/// Backward pass for F32/F16/BF16. Returns (dQ, dK, dV).
///
/// Two-step process:
/// 1. Preprocessing: compute D = rowsum(dO ⊙ O) per query position.
/// 2. Main backward kernel: compute dQ (atomicAdd), dK, dV.
pub(super) fn flash_attention_bwd_impl(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    causal: bool,
    window_size: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let dtype = q.dtype();

    let expected_o_shape = [p.batch_size, p.num_heads, p.seq_len_q, p.head_dim];
    if dout.shape() != expected_o_shape {
        return Err(Error::InvalidArgument {
            arg: "dout",
            reason: format!(
                "expected shape {:?}, got {:?}",
                expected_o_shape,
                dout.shape()
            ),
        });
    }
    if output.shape() != expected_o_shape {
        return Err(Error::InvalidArgument {
            arg: "output",
            reason: format!(
                "expected shape {:?}, got {:?}",
                expected_o_shape,
                output.shape()
            ),
        });
    }
    let expected_lse_shape = [p.batch_size, p.num_heads, p.seq_len_q];
    if lse.shape() != expected_lse_shape {
        return Err(Error::InvalidArgument {
            arg: "lse",
            reason: format!(
                "expected shape {:?}, got {:?}",
                expected_lse_shape,
                lse.shape()
            ),
        });
    }
    if !dout.is_contiguous() || !output.is_contiguous() || !lse.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg: "contiguity",
            reason: "backward requires contiguous dout, output, lse".into(),
        });
    }

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {:?} for flash_attention_bwd", dtype),
            });
        }
    };

    let device = q.device();
    let device_index = device.id();

    if window_size != 0 {
        return Err(Error::InvalidArgument {
            arg: "window_size",
            reason: "flash_attention_bwd: sliding window attention (window_size) is not yet \
                     supported in the backward pass; use window_size=0 for training"
                .into(),
        });
    }

    // Allocate gradient tensors (dQ must be zeroed — backward uses atomicAdd)
    let dq = Tensor::<CudaRuntime>::zeros(
        &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
        dtype,
        device,
    );
    let dk = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_kv_heads, p.seq_len_k, p.head_dim],
        dtype,
        device,
    );
    let dv = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_kv_heads, p.seq_len_k, p.head_dim],
        dtype,
        device,
    );

    // Step 1: Preprocessing — compute D = rowsum(dO ⊙ O) per query position
    // D shape: [B, num_heads, S_q]
    let d_buf = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q],
        DType::F32,
        device,
    );

    let module = kernels::get_or_load_module(client.context(), device_index, FLASH_V2_BWD_MODULE)?;

    {
        let preprocess_name = format!(
            "flash_attention_preprocess_bwd_{}_{}",
            p.head_dim, dtype_suffix
        );
        let func = kernels::get_kernel_function(&module, &preprocess_name)?;

        let block_size = 256u32;
        let grid_x = (p.batch_size * p.num_heads) as u32;
        let grid_y = (p.seq_len_q as u32).div_ceil(block_size);

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let dout_ptr = dout.ptr();
        let out_ptr = output.ptr();
        let d_ptr = d_buf.ptr();
        let batch_i32 = p.batch_size as i32;
        let nh_i32 = p.num_heads as i32;
        let sq_i32 = p.seq_len_q as i32;

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&dout_ptr);
            builder.arg(&out_ptr);
            builder.arg(&d_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention bwd preprocess failed: {:?}", e),
            })?;
        }
    }

    // Step 2: Main backward kernel — compute dQ, dK, dV
    {
        let sm_suffix = if p.use_sm_kernel { "_sm" } else { "" };
        let bwd_name = format!(
            "flash_attention_bwd_{}{}_{}",
            p.head_dim, sm_suffix, dtype_suffix
        );
        let func = kernels::get_kernel_function(&module, &bwd_name)?;

        // Shared memory: K[BLOCK_N][HD] + V[BLOCK_N][HD] + Q[BLOCK_M][HD] + dO[BLOCK_M][HD]
        let dtype_size = dtype.size_in_bytes();
        let smem_size = (2 * p.block_n * p.head_dim + 2 * p.block_m * p.head_dim) * dtype_size;
        set_smem_attribute(&func, smem_size)?;

        // Grid: (batch * num_heads, ceil(seq_len_k / BLOCK_N))
        let grid_x = (p.batch_size * p.num_heads) as u32;
        let grid_y = p.seq_len_k.div_ceil(p.block_n) as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (p.block_n as u32, 1, 1),
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
        let scale = (p.head_dim as f32).sqrt().recip();
        let batch_i32 = p.batch_size as i32;
        let nh_i32 = p.num_heads as i32;
        let sq_i32 = p.seq_len_q as i32;
        let sk_i32 = p.seq_len_k as i32;
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
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention bwd kernel launch failed: {:?}", e),
            })?;
        }
    }

    // Sync stream: BWD uses atomicAdd so must complete before results are read
    client
        .stream()
        .synchronize()
        .map_err(|e| Error::KernelError {
            reason: format!("Flash Attention bwd sync failed: {:?}", e),
        })?;

    Ok((dq, dk, dv))
}
