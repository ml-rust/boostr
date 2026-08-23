//! Flash Attention v2 FP8 backward pass: dQ, dK, dV with per-tensor scales.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FLASH_V2_BWD_MODULE};
use crate::ops::impl_generic::attention::sum_gqa_grads;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{AttentionParams, bwd_block_config, compute_bwd_smem, set_smem_attribute};

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

    if p.num_kv_heads == 0 {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: "num_kv_heads must be non-zero".into(),
        });
    }

    // GQA/MQA: the FP8 backward kernel takes no `num_kv_heads` — it indexes K, V,
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

    // Allocate gradient tensors (dQ zeroed for atomicAdd).
    // dK/dV are allocated with `num_heads` heads to match the kernel's indexing;
    // GQA groups are summed back to `num_kv_heads` after the launch.
    //
    // dQ is an F32 accumulator, not an FP8 buffer: each K/V block adds into the
    // same dQ element with `atomicAdd`, and CUDA has no 1-byte float atomic. A
    // 4-byte atomic aimed at an FP8 element is misaligned unless the element
    // index is a multiple of 4 (`CUDA_ERROR_MISALIGNED_ADDRESS`) and clobbers
    // three neighbours when it is. The kernel therefore accumulates the
    // DEQUANTIZED gradient here; `dq_scale` and the quantization are applied
    // below, matching the `raw = quantize(value * scale)` convention the kernel
    // uses for dK/dV.
    let dq_acc = Tensor::<CudaRuntime>::zeros(
        &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
        DType::F32,
        device,
    );
    let dk = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_k, p.head_dim],
        dtype,
        device,
    );
    let dv = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, p.seq_len_k, p.head_dim],
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
        // The backward layout needs more shared memory than the forward one that
        // `AttentionParams` was sized for, so pick a backward-specific block config
        // that fits this device; `_sm` selects the small-block instantiations.
        // FP8 is 1 byte per element.
        let (block_m, block_n, use_sm_kernel) = bwd_block_config(p.head_dim, 1)?;
        let sm_infix = if use_sm_kernel { "_sm" } else { "" };
        let bwd_name = format!("flash_attention_bwd_{}{}_fp8", p.head_dim, sm_infix);
        let func = kernels::get_kernel_function(&module, &bwd_name)?;

        let smem_size = compute_bwd_smem(block_m, block_n, p.head_dim, 1);
        set_smem_attribute(&func, smem_size)?;

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

    // Sync before the requantization below: the backward uses atomicAdd, and
    // `dq_acc` is a temporary that must outlive the kernel writing into it.
    client
        .stream()
        .synchronize()
        .map_err(|e| Error::KernelError {
            reason: format!("Flash Attention FP8 bwd sync failed: {:?}", e),
        })?;

    // Requantize dQ once: the kernel stores `raw = quantize(value * scale)`, so
    // apply `dq_scale` to the F32 accumulator before the single cast to FP8.
    let dq_scaled = client.mul_scalar(&dq_acc, f64::from(dq_scale))?;
    let dq = client.cast(&dq_scaled, dtype)?;

    if repeats > 1 {
        let dk = sum_gqa_grads_fp8(client, &dk, p.num_kv_heads, repeats, dtype)?;
        let dv = sum_gqa_grads_fp8(client, &dv, p.num_kv_heads, repeats, dtype)?;
        return Ok((dq, dk, dv));
    }

    Ok((dq, dk, dv))
}

/// Reduce a GQA group of per-head FP8 gradients back to one KV head.
///
/// The kernel stores `raw = quantize(value * scale)`, so an element already
/// carries its `dk_scale`/`dv_scale` factor. Summing raw FP8 would round once per
/// group member and can leave E4M3's ~±448 range, so the group is dequantized to
/// F32, summed there, and requantized ONCE. The cast back to FP8 reapplies the
/// kernel's convention exactly: the F32 sum equals `scale * sum(real values)`,
/// which is the value the kernel would have written for the merged head.
fn sum_gqa_grads_fp8(
    client: &CudaClient,
    grad: &Tensor<CudaRuntime>,
    num_kv_heads: usize,
    repeats: usize,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let grad_f32 = client.cast(grad, DType::F32)?;
    let summed = sum_gqa_grads(client, &grad_f32, num_kv_heads, repeats)?;
    Ok(client.cast(&summed, dtype)?)
}
