//! Flash Attention v2 backward pass (F32/F16/BF16): dQ, dK, dV computation.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FLASH_V2_BWD_MODULE};
use crate::ops::impl_generic::attention::sum_gqa_grads;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::{ShapeOps, TypeConversionOps};
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{AttentionParams, bwd_block_config, compute_bwd_smem, set_smem_attribute};

/// Backward pass for F32/F16/BF16. Returns (dQ, dK, dV).
///
/// Steps:
/// 0. GQA/MQA: repeat KV heads up to `num_heads` (the kernel has no
///    `num_kv_heads`), and sum the per-head dK/dV back down afterwards.
/// 1. Preprocessing: compute D = rowsum(dO ⊙ O) per query position.
/// 2. Main backward kernel: compute dQ (atomicAdd), dK, dV.
#[allow(clippy::too_many_arguments)]
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

    if p.num_kv_heads == 0 {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: "num_kv_heads must be non-zero".into(),
        });
    }

    // GQA/MQA: the v2 backward kernel takes no `num_kv_heads` — it indexes K, V,
    // dK and dV with `num_heads`. Repeat the KV heads up to `num_heads` (same
    // mapping the forward kernel applies internally:
    // `kv_head_idx = head_idx / (num_heads / num_kv_heads)`), run the kernel over
    // the expanded layout, then SUM each KV head's group of per-head dK/dV back
    // down to `num_kv_heads`. Sum is the gradient of a repeated tensor.
    let repeats = p.num_heads / p.num_kv_heads;
    let kv_repeated = if repeats > 1 {
        let k_rep = client
            .repeat_interleave(k, repeats, Some(1))?
            .contiguous()?;
        let v_rep = client
            .repeat_interleave(v, repeats, Some(1))?
            .contiguous()?;
        Some((k_rep, v_rep))
    } else {
        None
    };
    let (k, v) = match &kv_repeated {
        Some((k_rep, v_rep)) => (k_rep, v_rep),
        None => (k, v),
    };

    // Allocate gradient tensors (dQ must be zeroed — backward uses atomicAdd).
    // dK/dV are allocated with `num_heads` heads to match the kernel's indexing;
    // GQA groups are summed back to `num_kv_heads` after the launch.
    //
    // dQ is ALWAYS F32, whatever `dtype` is: each K/V block adds into the same
    // dQ element with `atomicAdd`, and CUDA has no 2-byte float atomic. A
    // 4-byte atomic aimed at an F16/BF16 element is misaligned whenever the
    // element index is odd (`CUDA_ERROR_MISALIGNED_ADDRESS`) and clobbers the
    // neighbouring element when it is even. The F32 accumulator is cast down to
    // `dtype` after the kernel completes.
    let dq_acc = Tensor::<CudaRuntime>::zeros(
        &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
        DType::F32,
        device,
    )?;
    let dk = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_k, p.head_dim],
        dtype,
        device,
    )?;
    let dv = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_k, p.head_dim],
        dtype,
        device,
    )?;

    // Step 1: Preprocessing — compute D = rowsum(dO ⊙ O) per query position
    // D shape: [B, num_heads, S_q]
    let d_buf = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_q],
        DType::F32,
        device,
    )?;

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
        // The backward layout needs more shared memory than the forward one that
        // `AttentionParams` was sized for, so pick a backward-specific block config
        // that fits this device; `_sm` selects the small-block instantiations.
        let dtype_size = dtype.size_in_bytes();
        let (block_m, block_n, use_sm_kernel) = bwd_block_config(p.head_dim, dtype_size)?;
        let sm_infix = if use_sm_kernel { "_sm" } else { "" };
        let bwd_name = format!(
            "flash_attention_bwd_{}{}_{}",
            p.head_dim, sm_infix, dtype_suffix
        );
        let func = kernels::get_kernel_function(&module, &bwd_name)?;

        // Shared memory: K[BLOCK_N][HD] + V[BLOCK_N][HD] + Q[BLOCK_M][HD] + dO[BLOCK_M][HD]
        let smem_size = compute_bwd_smem(block_m, block_n, p.head_dim, dtype_size);
        set_smem_attribute(&func, smem_size)?;

        // Grid: (batch * num_heads, ceil(seq_len_k / BLOCK_N))
        let grid_x = (p.batch_size * p.num_heads) as u32;
        let grid_y = p.seq_len_k.div_ceil(block_n) as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_n as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let o_ptr = output.ptr();
        let dout_ptr = dout.ptr();
        let lse_ptr = lse.ptr();
        let d_ptr = d_buf.ptr();
        let dq_ptr = dq_acc.ptr();
        let dk_ptr = dk.ptr();
        let dv_ptr = dv.ptr();
        let scale = (p.head_dim as f32).sqrt().recip();
        let batch_i32 = p.batch_size as i32;
        let nh_i32 = p.num_heads as i32;
        let sq_i32 = p.seq_len_q as i32;
        let sk_i32 = p.seq_len_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };
        let window_i32 = i32::try_from(window_size).map_err(|_| Error::InvalidArgument {
            arg: "window_size",
            reason: format!("window_size {} exceeds i32 range", window_size),
        })?;

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
            builder.arg(&window_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention bwd kernel launch failed: {:?}", e),
            })?;
        }
    }

    // Narrow the F32 dQ accumulator back to the caller's dtype. No stream sync
    // first: `cast` launches on the same `CudaClient` stream as the kernel above,
    // and a stream runs in issue order, so the cast cannot read `dq_acc` before
    // the atomics into it have retired. Same shape as `mqa_gqa/bwd.rs`.
    let dq = if dtype == DType::F32 {
        dq_acc
    } else {
        client.cast(&dq_acc, dtype)?
    };

    if repeats > 1 {
        let dk = sum_gqa_grads(client, &dk, p.num_kv_heads, repeats)?;
        let dv = sum_gqa_grads(client, &dv, p.num_kv_heads, repeats)?;
        return Ok((dq, dk, dv));
    }

    Ok((dq, dk, dv))
}
