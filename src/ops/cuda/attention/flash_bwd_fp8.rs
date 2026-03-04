//! Flash Attention v2 FP8 backward pass: dQ, dK, dV with per-tensor scales.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FLASH_V2_BWD_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{AttentionParams, set_smem_attribute};

/// FP8 backward pass (E4M3/E5M2). Returns (dQ, dK, dV).
///
/// Two-step process with extra per-tensor scale args:
/// 1. FP8 preprocessing: D = rowsum(dO ⊙ O) with do_scale and o_scale.
/// 2. FP8 main backward kernel with full scale set.
#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_bwd_fp8_impl(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    causal: bool,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    do_scale: f32,
    o_scale: f32,
    dq_scale: f32,
    dk_scale: f32,
    dv_scale: f32,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let dtype = q.dtype();

    let expected_o_shape = [p.batch_size, p.num_heads, p.seq_len_q, p.head_dim];
    if dout.shape() != expected_o_shape || output.shape() != expected_o_shape {
        return Err(Error::InvalidArgument {
            arg: "dout/output",
            reason: format!("expected shape {:?}", expected_o_shape),
        });
    }
    let expected_lse_shape = [p.batch_size, p.num_heads, p.seq_len_q];
    if lse.shape() != expected_lse_shape {
        return Err(Error::InvalidArgument {
            arg: "lse",
            reason: format!("expected shape {:?}", expected_lse_shape),
        });
    }

    let device = q.device();
    let device_index = device.id();

    // Allocate gradient tensors (dQ zeroed for atomicAdd)
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

    let d_buf = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q],
        DType::F32,
        device,
    );

    let module = kernels::get_or_load_module(client.context(), device_index, FLASH_V2_BWD_MODULE)?;

    // Step 1: FP8 Preprocessing — extra scale args
    {
        let preprocess_name = format!("flash_attention_preprocess_bwd_{}_fp8", p.head_dim);
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
            builder.arg(&do_scale);
            builder.arg(&o_scale);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention FP8 bwd preprocess failed: {:?}", e),
            })?;
        }
    }

    // Step 2: FP8 Main backward — extra scale args
    {
        let sm_suffix = if p.use_sm_kernel { "_sm" } else { "" };
        let bwd_name = format!("flash_attention_bwd_{}{}_fp8", p.head_dim, sm_suffix);
        let func = kernels::get_kernel_function(&module, &bwd_name)?;

        // FP8 is 1 byte per element
        let smem_size = 2 * p.block_n * p.head_dim + 2 * p.block_m * p.head_dim;
        set_smem_attribute(&func, smem_size)?;

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
            builder.arg(&q_scale);
            builder.arg(&k_scale);
            builder.arg(&v_scale);
            builder.arg(&do_scale);
            builder.arg(&dq_scale);
            builder.arg(&dk_scale);
            builder.arg(&dv_scale);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention FP8 bwd kernel launch failed: {:?}", e),
            })?;
        }
    }

    Ok((dq, dk, dv))
}
