//! VarLen (ragged) attention backward CUDA launcher.
//!
//! Split out of `varlen_attention.rs` to keep that file's `VarLenAttentionOps`
//! trait impl as wiring only — mirrors `paged_attention_bwd.rs`'s split from
//! `paged_attention.rs`.

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash::impl_ops::set_smem_attribute;
use super::varlen_attention_block_config::{block_config_with_override, bwd_smem_size};
use crate::ops::cuda::kernels::{
    self, VARLEN_ATTENTION_BWD_FP16_MODULE, VARLEN_ATTENTION_BWD_MODULE,
};

/// Production entry point: normal capability-gated tile selection (no override).
#[allow(clippy::too_many_arguments)]
pub(super) fn varlen_attention_bwd_impl(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    cu_seqlens_q: &Tensor<CudaRuntime>,
    cu_seqlens_k: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    head_dim: usize,
    causal: bool,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    varlen_attention_bwd_impl_inner(
        client,
        dout,
        q,
        k,
        v,
        output,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
        causal,
        None,
    )
}

/// Test-only entry point: run varlen attention backward with an explicit
/// large/small tile choice instead of the normal capability gate. Rust test
/// binaries run multi-threaded in one process, so forcing the tile via a
/// process-wide env var would race with every other test; parity tests call
/// this instead to force each side safely. `force_large` is refused (with an
/// error, not a silent fallback) when the requested tile does not exist or
/// does not fit this device — see [`block_config_with_override`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn varlen_attention_bwd_with_tile_for_test(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    cu_seqlens_q: &Tensor<CudaRuntime>,
    cu_seqlens_k: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    head_dim: usize,
    causal: bool,
    force_large: bool,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    varlen_attention_bwd_impl_inner(
        client,
        dout,
        q,
        k,
        v,
        output,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
        causal,
        Some(force_large),
    )
}

#[allow(clippy::too_many_arguments)]
fn varlen_attention_bwd_impl_inner(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    cu_seqlens_q: &Tensor<CudaRuntime>,
    cu_seqlens_k: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    head_dim: usize,
    causal: bool,
    force_large: Option<bool>,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    if head_dim != 64 && head_dim != 128 && head_dim != 256 {
        return Err(Error::KernelError {
            reason: format!(
                "varlen attention bwd: unsupported head_dim {head_dim}, only 64/128/256"
            ),
        });
    }

    let dtype = q.dtype();
    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        _ => {
            return Err(Error::KernelError {
                reason: format!("varlen attention bwd: unsupported dtype {dtype:?}"),
            });
        }
    };

    // FP16 backward kernels live in their own compiled module (split out to
    // keep each .cu within the file-size budget); FP32 stays in the base module.
    let bwd_module = match dtype {
        DType::F16 => VARLEN_ATTENTION_BWD_FP16_MODULE,
        _ => VARLEN_ATTENTION_BWD_MODULE,
    };

    let device = q.device();
    let device_index = device.id();

    // Same tile-selection call as fwd (sized on the backward requirement, the
    // binding constraint) so a given (head_dim, dtype) call site is internally
    // consistent — see `varlen_attention_block_config::block_config_with_override`.
    let (block_m, block_n, variant) = block_config_with_override(head_dim, dtype, force_large)?;
    let tile_suffix = variant.suffix();
    let kernel_name = format!("varlen_flash_attention_bwd_{head_dim}_{dtype_suffix}{tile_suffix}");

    let module = kernels::get_or_load_module(client.context(), device_index, bwd_module)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let total_tokens_q = q.shape()[0];
    let total_tokens_k = k.shape()[0];

    // dq: same head layout as Q (num_heads)
    // dk/dv: kv head layout (num_kv_heads) — GQA: fewer heads than Q
    let dq = Tensor::<CudaRuntime>::zeros(&[total_tokens_q, num_heads, head_dim], dtype, device)?;
    let dk =
        Tensor::<CudaRuntime>::zeros(&[total_tokens_k, num_kv_heads, head_dim], dtype, device)?;
    let dv =
        Tensor::<CudaRuntime>::zeros(&[total_tokens_k, num_kv_heads, head_dim], dtype, device)?;

    let num_q_blocks_per_batch = max_seqlen_q.div_ceil(block_m);
    let num_q_blocks = num_q_blocks_per_batch * batch_size;

    // Shared memory layout (with +1 HEAD_STRIDE padding, same as the bwd kernel):
    //   Q tile   : BLOCK_M * (HEAD_DIM+1) elements
    //   K tile   : BLOCK_N * (HEAD_DIM+1) elements
    //   V tile   : BLOCK_N * (HEAD_DIM+1) elements
    //   dO tile  : BLOCK_M * (HEAD_DIM+1) elements
    // Total: (2*BLOCK_M + 2*BLOCK_N) * HEAD_STRIDE * dtype_size bytes
    let dtype_size = dtype.size_in_bytes();
    let smem_size = bwd_smem_size(block_m, block_n, head_dim, dtype_size);
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: ((num_q_blocks * num_heads) as u32, 1, 1),
        block_dim: (block_m as u32, 1, 1),
        shared_mem_bytes: smem_size as u32,
    };

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let l_ptr = lse.ptr();
    let do_ptr = dout.ptr();
    let cu_q_ptr = cu_seqlens_q.ptr();
    let cu_k_ptr = cu_seqlens_k.ptr();
    let dq_ptr = dq.ptr();
    let dk_ptr = dk.ptr();
    let dv_ptr = dv.ptr();
    let scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let nkv_i32 = num_kv_heads as i32;
    let msq_i32 = max_seqlen_q as i32;
    let msk_i32 = max_seqlen_k as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&l_ptr);
        builder.arg(&do_ptr);
        builder.arg(&cu_q_ptr);
        builder.arg(&cu_k_ptr);
        builder.arg(&dq_ptr);
        builder.arg(&dk_ptr);
        builder.arg(&dv_ptr);
        builder.arg(&batch_i32);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&msq_i32);
        builder.arg(&msk_i32);
        builder.arg(&scale);
        builder.arg(&causal_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("varlen attention bwd launch failed: {e:?}"),
        })?;
    }

    // Sync stream: BWD uses atomicAdd so must complete before results are read
    client
        .stream()
        .synchronize()
        .map_err(|e| Error::KernelError {
            reason: format!("varlen attention bwd sync failed: {e:?}"),
        })?;

    Ok((dq, dk, dv))
}
