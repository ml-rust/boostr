//! CUDA launcher for `flash_attention_fwd_fp8_kv` — Flash Attention over an
//! FP8-quantized KV cache.
//!
//! Kernel: `kv_cache_quant.cu`, `flash_attention_fp8_kv_impl<HEAD_DIM, BLOCK_M,
//! BLOCK_N>`, entry points `flash_attention_fp8_kv_64`/`_128` (both
//! BLOCK_M=128, BLOCK_N=64). Q stays F32; only K and V are FP8 E4M3. Unlike
//! `flash_v2_fp8.cu`, K/V scales are per-token or per-head TENSORS, and the
//! kernel takes no `num_kv_heads` — it indexes K/V with `num_heads` directly,
//! so it has no GQA path.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_QUANT_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{device_max_smem, set_smem_attribute};

const BLOCK_M: usize = 128;
const BLOCK_N: usize = 64;

/// Shared memory `flash_attention_fp8_kv_impl` needs:
/// `(BLOCK_M + 2*BLOCK_N) * HEAD_DIM * sizeof(f32)`. K and V are dequantized
/// into F32 shared-memory tiles, so the byte size uses `f32`, not the FP8
/// element size. No bank-conflict `+1` padding, unlike `flash_smem::compute_smem`
/// — `Q_smem`/`K_smem`/`V_smem` in `kv_cache_quant.cu` pack `HEAD_DIM` tight.
fn fp8_kv_smem_bytes(head_dim: usize) -> usize {
    (BLOCK_M + 2 * BLOCK_N) * head_dim * std::mem::size_of::<f32>()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_fwd_fp8_kv_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_quant: &Tensor<CudaRuntime>,
    v_quant: &Tensor<CudaRuntime>,
    k_scales: &Tensor<CudaRuntime>,
    v_scales: &Tensor<CudaRuntime>,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
    per_token_scales: bool,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    if q.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!(
                "flash_attention_fwd_fp8_kv requires Q in F32, got {:?}",
                q.dtype()
            ),
        });
    }
    for (name, t) in [("k_quant", k_quant), ("v_quant", v_quant)] {
        if t.dtype() != DType::FP8E4M3 {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "flash_attention_fwd_fp8_kv requires {name} in FP8E4M3, got {:?}",
                    t.dtype()
                ),
            });
        }
    }

    let kernel_name = match head_dim {
        64 => "flash_attention_fp8_kv_64",
        128 => "flash_attention_fp8_kv_128",
        other => {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "flash_attention_fwd_fp8_kv supports head_dim 64 or 128, got {other}"
                ),
            });
        }
    };

    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    let (batch_size, q_heads, seq_len_q, q_head_dim) =
        (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    if q_heads != num_heads || q_head_dim != head_dim {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!(
                "q shape [{batch_size}, {q_heads}, {seq_len_q}, {q_head_dim}] does not match \
                 num_heads={num_heads} head_dim={head_dim}"
            ),
        });
    }

    let k_shape = k_quant.shape().to_vec();
    if k_shape.len() != 4
        || k_shape[0] != batch_size
        || k_shape[1] != num_heads
        || k_shape[3] != head_dim
    {
        return Err(Error::InvalidArgument {
            arg: "k_quant",
            reason: format!(
                "expected [{batch_size}, {num_heads}, seq_len_k, {head_dim}], got {k_shape:?}"
            ),
        });
    }
    let seq_len_k = k_shape[2];
    if v_quant.shape() != k_shape.as_slice() {
        return Err(Error::InvalidArgument {
            arg: "v_quant",
            reason: format!(
                "v_quant shape {:?} must match k_quant shape {:?}",
                v_quant.shape(),
                k_shape
            ),
        });
    }

    let expected_scale_len = if per_token_scales {
        batch_size * num_heads * seq_len_k
    } else {
        batch_size * num_heads
    };
    for (name, s) in [("k_scales", k_scales), ("v_scales", v_scales)] {
        let n: usize = s.shape().iter().product();
        if n != expected_scale_len {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "expected {expected_scale_len} {} scales, got {n}",
                    if per_token_scales {
                        "per-token"
                    } else {
                        "per-head"
                    }
                ),
            });
        }
    }

    // Check the shared-memory requirement before launching: the default CUDA
    // cap is 48KB per block, and both head dims here need more than that
    // (64KB for head_dim=64, 128KB for head_dim=128), so `set_smem_attribute`
    // always takes the opt-in path below. Naming the head dim here gives a
    // clearer error than its generic message would.
    let smem_size = fp8_kv_smem_bytes(head_dim);
    let max_smem = device_max_smem();
    if smem_size > max_smem {
        return Err(Error::KernelError {
            reason: format!(
                "flash_attention_fwd_fp8_kv: head_dim={head_dim} needs {}KB shared memory, \
                 device limit is {}KB",
                smem_size / 1024,
                max_smem / 1024
            ),
        });
    }

    let device = q.device();
    let output = Tensor::<CudaRuntime>::empty(
        &[batch_size, num_heads, seq_len_q, head_dim],
        DType::F32,
        device,
    )?;
    let lse =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, KV_CACHE_QUANT_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (batch_size * num_heads) as u32,
            seq_len_q.div_ceil(BLOCK_M) as u32,
            1,
        ),
        block_dim: (BLOCK_M as u32, 1, 1),
        shared_mem_bytes: smem_size as u32,
    };

    let q_ptr = q.ptr();
    let k_ptr = k_quant.ptr();
    let v_ptr = v_quant.ptr();
    let ks_ptr = k_scales.ptr();
    let vs_ptr = v_scales.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let sq_i32 = seq_len_q as i32;
    let sk_i32 = seq_len_k as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };
    let per_token_i32 = if per_token_scales { 1i32 } else { 0i32 };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&ks_ptr);
        builder.arg(&vs_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&sq_i32);
        builder.arg(&sk_i32);
        builder.arg(&scale);
        builder.arg(&causal_i32);
        builder.arg(&per_token_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Flash Attention FP8-KV fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
