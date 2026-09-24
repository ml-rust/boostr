//! Standard (non-FP8) paged attention forward kernel launcher.
//!
//! `seq_len_q == 1` in a decode dtype at head dim 64 or 128 takes the decode
//! fast path, the only shape here that honors a per-row `KvSeqLen::PerRow`
//! key end. Every tiled shape refuses it.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, PAGED_ATTENTION_MODULE};
use crate::ops::impl_generic::attention::{per_row_kv_seq_len_unsupported, validate_row_bounds};
use crate::ops::traits::KvSeqLen;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::decode_split::decode_supports_dtype;
use super::flash::flash_utils::set_smem_attribute;
use super::paged_attention_fwd_block_config::{
    fwd_block_config, fwd_block_config_with_override, fwd_smem_size,
};
use super::paged_decode::paged_decode_attention_fwd;

/// Standard (non-FP8) paged attention forward.
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_attention_fwd_impl(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
    kv_seq_len: Option<KvSeqLen<'_, CudaRuntime>>,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    paged_attention_fwd_impl_inner(
        client,
        q,
        k_blocks,
        v_blocks,
        block_table,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_size,
        causal,
        kv_seq_len,
        None,
    )
}

/// Test-only entry point: run the standard (non-FP8) paged attention prefill
/// forward with an explicit large/small tile choice instead of reading
/// `BOOSTR_PAGED_PREFILL_TILE`. Rust test binaries run multi-threaded in one
/// process, so setting that process-wide env var from a test would race with
/// every other test reading or setting it; parity tests call this instead to
/// force each side safely. Has no effect on tile choice when `seq_len_q == 1`
/// — that shape always takes the specialized decode fast path, which has no
/// large/small tile of its own.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_fwd_with_tile_for_test(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
    force_large: bool,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    // Forcing a prefill tile only makes sense above one query row, the shape
    // that refuses a per-row key end anyway.
    paged_attention_fwd_impl_inner(
        client,
        q,
        k_blocks,
        v_blocks,
        block_table,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        block_size,
        causal,
        None,
        Some(force_large),
    )
}

#[allow(clippy::too_many_arguments)]
fn paged_attention_fwd_impl_inner(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_blocks: &Tensor<CudaRuntime>,
    v_blocks: &Tensor<CudaRuntime>,
    block_table: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
    kv_seq_len: Option<KvSeqLen<'_, CudaRuntime>>,
    force_large: Option<bool>,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    let batch_size = q_shape[0];
    let dtype = q.dtype();

    // Fast path: S_q=1 decode with specialized kernel (no shared memory tiling overhead)
    if seq_len_q == 1 && decode_supports_dtype(dtype) && (head_dim == 64 || head_dim == 128) {
        // Per-row ends reach the decode kernel as a device pointer; null means
        // every row ends at `seq_len_k`.
        let kv_end_ptr = match kv_seq_len {
            Some(KvSeqLen::PerRow(t)) => {
                validate_row_bounds(t, batch_size, "kv_seq_len")?;
                t.ptr()
            }
            _ => 0,
        };
        return paged_decode_attention_fwd(
            client,
            q,
            k_blocks,
            v_blocks,
            block_table,
            batch_size,
            num_heads,
            num_kv_heads,
            seq_len_k,
            head_dim,
            block_size,
            kv_end_ptr,
        );
    }

    // Every shape below reads one key length for the whole batch.
    if let Some(KvSeqLen::PerRow(_)) = kv_seq_len {
        return Err(per_row_kv_seq_len_unsupported("CUDA"));
    }

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "unsupported dtype {:?}. Use paged_attention_fwd_fp8 for FP8.",
                    dtype
                ),
            });
        }
    };

    let device = q.device();
    let device_index = device.id();
    let (block_m, block_n, use_large) = match force_large {
        Some(forced) => fwd_block_config_with_override(
            head_dim,
            dtype,
            seq_len_q,
            num_heads,
            batch_size,
            device_index,
            Some(forced),
        )?,
        None => fwd_block_config(
            head_dim,
            dtype,
            seq_len_q,
            num_heads,
            batch_size,
            device_index,
        )?,
    };
    let tile_suffix = if use_large { "" } else { "_small" };
    let kernel_name = format!(
        "paged_flash_attention_fwd_{}_{}{}",
        head_dim, dtype_suffix, tile_suffix
    );

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device)?;
    let lse =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    let dtype_size = dtype.size_in_bytes();
    let smem_size = fwd_smem_size(block_m, block_n, head_dim, dtype_size);

    let max_num_blocks = block_table.shape()[1];
    let module =
        kernels::get_or_load_module(client.context(), device_index, PAGED_ATTENTION_MODULE)?;
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
    let kb_ptr = k_blocks.ptr();
    let vb_ptr = v_blocks.ptr();
    let bt_ptr = block_table.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
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
        builder.arg(&bt_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
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
            reason: format!("Paged attention fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
