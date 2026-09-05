//! CUDA launcher for `flash_attention_fwd_int4_kv` — Flash Attention over an
//! INT4-quantized KV cache.
//!
//! Kernel: `kv_cache_int4.cu`, `flash_attention_int4_kv_impl<HEAD_DIM,
//! BLOCK_M, BLOCK_N>`, entry points `flash_attention_int4_kv_64`/`_128` (large
//! tile) and `_64_small`/`_128_small` (fallback tile for devices that can't
//! fit the large tile's shared memory). Q stays F32; K/V are packed INT4 with
//! per-group asymmetric (scale, zero) in F16. Same large/`_small` fallback
//! pattern as `flash_fwd_fp8_kv.rs` — tries the large tile first and falls
//! back to `_small` when it doesn't fit — same module as `append_kv_int4`
//! (`KV_CACHE_INT4_MODULE`), reused rather than re-registered.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_INT4_MODULE};
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_utils::{device_max_smem, set_smem_attribute};

/// (BLOCK_M, BLOCK_N) for the large and `_small` kernel variants, keyed by
/// head_dim. Must stay in sync with the `extern "C"` instantiations in
/// `kv_cache_int4.cu`.
fn block_config(head_dim: usize) -> Result<((usize, usize), (usize, usize))> {
    match head_dim {
        64 => Ok(((128, 64), (64, 32))),
        128 => Ok(((64, 64), (32, 32))),
        other => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!("flash_attention_fwd_int4_kv supports head_dim 64 or 128, got {other}"),
        }),
    }
}

/// Shared memory `flash_attention_int4_kv_impl` needs:
/// `(BLOCK_M + 2*BLOCK_N) * HEAD_DIM * sizeof(f32)`. K and V are dequantized
/// into F32 shared-memory tiles. No bank-conflict `+1` padding, unlike
/// `flash_smem::compute_smem` — `Q_smem`/`K_smem`/`V_smem` in
/// `kv_cache_int4.cu` pack `HEAD_DIM` tight (same layout as
/// `fp8_kv_smem_bytes` in `flash_fwd_fp8_kv.rs`).
fn int4_kv_smem_bytes(block_m: usize, block_n: usize, head_dim: usize) -> usize {
    (block_m + 2 * block_n) * head_dim * std::mem::size_of::<f32>()
}

/// Pick the large or `_small` kernel variant for `head_dim`, based on which
/// fits this device's opt-in shared-memory limit. Returns
/// `(kernel_name, block_m, block_n, smem_bytes)`.
fn select_kernel(head_dim: usize) -> Result<(&'static str, usize, usize, usize)> {
    let ((large_m, large_n), (small_m, small_n)) = block_config(head_dim)?;
    let max_smem = device_max_smem();

    let large_smem = int4_kv_smem_bytes(large_m, large_n, head_dim);
    if large_smem <= max_smem {
        let name = match head_dim {
            64 => "flash_attention_int4_kv_64",
            128 => "flash_attention_int4_kv_128",
            _ => unreachable!("block_config already validated head_dim"),
        };
        return Ok((name, large_m, large_n, large_smem));
    }

    let small_smem = int4_kv_smem_bytes(small_m, small_n, head_dim);
    if small_smem <= max_smem {
        let name = match head_dim {
            64 => "flash_attention_int4_kv_64_small",
            128 => "flash_attention_int4_kv_128_small",
            _ => unreachable!("block_config already validated head_dim"),
        };
        return Ok((name, small_m, small_n, small_smem));
    }

    Err(Error::KernelError {
        reason: format!(
            "flash_attention_fwd_int4_kv: head_dim={head_dim} needs at least {}KB shared \
             memory even for the small tile, device limit is {}KB",
            small_smem / 1024,
            max_smem / 1024
        ),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_fwd_int4_kv_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_quant: &Tensor<CudaRuntime>,
    v_quant: &Tensor<CudaRuntime>,
    k_scales: &Tensor<CudaRuntime>,
    k_zeros: &Tensor<CudaRuntime>,
    v_scales: &Tensor<CudaRuntime>,
    v_zeros: &Tensor<CudaRuntime>,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
    group_size: Int4GroupSize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    if q.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!(
                "flash_attention_fwd_int4_kv requires Q in F32, got {:?}",
                q.dtype()
            ),
        });
    }
    for (name, t) in [("k_quant", k_quant), ("v_quant", v_quant)] {
        if t.dtype() != DType::U8 {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "flash_attention_fwd_int4_kv requires {name} in U8, got {:?}",
                    t.dtype()
                ),
            });
        }
    }
    for (name, t) in [
        ("k_scales", k_scales),
        ("k_zeros", k_zeros),
        ("v_scales", v_scales),
        ("v_zeros", v_zeros),
    ] {
        if t.dtype() != DType::F16 {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "flash_attention_fwd_int4_kv requires {name} in F16, got {:?}",
                    t.dtype()
                ),
            });
        }
    }

    // The CUDA kernel groups int4 elements by `elem_col / group_size` WITHIN
    // a token (`kv_cache_int4.cu`, `flash_attention_int4_kv_impl`). The CPU
    // reference (`dequantize_kv_int4`) groups by flattened index `i /
    // group_size` over `token*head_dim`. The two conventions agree only when
    // `head_dim % group_size == 0` — otherwise groups straddle token
    // boundaries on CPU but never on CUDA, and the backends silently
    // disagree. Do not remove this check to "simplify": it is the only thing
    // stopping that silent divergence.
    let gs = group_size as usize;
    if !head_dim.is_multiple_of(gs) {
        return Err(Error::InvalidArgument {
            arg: "group_size",
            reason: format!(
                "flash_attention_fwd_int4_kv requires head_dim ({head_dim}) to be a multiple \
                 of group_size ({gs}): per-token grouping only agrees with the CPU reference's \
                 flattened grouping when the group divides head_dim evenly"
            ),
        });
    }

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
        || k_shape[3] != head_dim / 2
    {
        return Err(Error::InvalidArgument {
            arg: "k_quant",
            reason: format!(
                "expected [{batch_size}, {num_heads}, seq_len_k, {}], got {k_shape:?}",
                head_dim / 2
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

    let groups_per_token = head_dim / gs;
    let required_scale_elems = batch_size * num_heads * seq_len_k * groups_per_token;
    for (name, t) in [
        ("k_scales", k_scales),
        ("k_zeros", k_zeros),
        ("v_scales", v_scales),
        ("v_zeros", v_zeros),
    ] {
        // The kernel indexes scales/zeros up to `required_scale_elems`. An
        // undersized tensor reads past its allocation on the device.
        if t.numel() < required_scale_elems {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "flash_attention_fwd_int4_kv needs {required_scale_elems} elements, got {}",
                    t.numel()
                ),
            });
        }
    }

    let (kernel_name, block_m, _block_n, smem_size) = select_kernel(head_dim)?;

    let device = q.device();
    let output = Tensor::<CudaRuntime>::empty(
        &[batch_size, num_heads, seq_len_q, head_dim],
        DType::F32,
        device,
    )?;
    let lse =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_CACHE_INT4_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;
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
    let k_ptr = k_quant.ptr();
    let v_ptr = v_quant.ptr();
    let ks_ptr = k_scales.ptr();
    let kz_ptr = k_zeros.ptr();
    let vs_ptr = v_scales.ptr();
    let vz_ptr = v_zeros.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let sq_i32 = seq_len_q as i32;
    let sk_i32 = seq_len_k as i32;
    let gs_i32 = gs as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&ks_ptr);
        builder.arg(&kz_ptr);
        builder.arg(&vs_ptr);
        builder.arg(&vz_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&sq_i32);
        builder.arg(&sk_i32);
        builder.arg(&gs_i32);
        builder.arg(&scale);
        builder.arg(&causal_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("Flash Attention INT4-KV fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
