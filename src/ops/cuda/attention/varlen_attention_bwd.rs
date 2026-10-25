//! VarLen (ragged) attention backward CUDA launcher.
//!
//! `varlen_attention.rs` holds the `VarLenAttentionOps` trait wiring.

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash::impl_ops::set_smem_attribute;
use super::varlen_attention_block_config::{
    block_config_with_override, bwd_smem_size, validate_head_dim,
};
use super::varlen_attention_check::{VarlenDims, check_bwd_args};
use super::varlen_attention_fwd::window_size_arg;
use crate::ops::cuda::kernels::{
    self, VARLEN_ATTENTION_BWD_HALF_MODULE, VARLEN_ATTENTION_BWD_MODULE,
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
    window_size: usize,
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
        window_size,
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
    window_size: usize,
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
        window_size,
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
    window_size: usize,
    force_large: Option<bool>,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    validate_head_dim(head_dim, "varlen attention bwd")?;
    let dims = VarlenDims {
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
    };
    let totals = check_bwd_args(dout, q, k, v, output, lse, cu_seqlens_q, cu_seqlens_k, dims)?;
    // The kernels read dense rows: strided inputs go in as contiguous copies.
    let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
    let (dout, output, lse) = (dout.contiguous()?, output.contiguous()?, lse.contiguous()?);
    let cu_seqlens_q = cu_seqlens_q.contiguous()?;
    let cu_seqlens_k = cu_seqlens_k.contiguous()?;

    let dtype = q.dtype();
    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::KernelError {
                reason: format!(
                    "varlen attention bwd: unsupported dtype {dtype:?}, only F32/F16/BF16; \
                     cast q/k/v to F32, F16 or BF16"
                ),
            });
        }
    };

    // F16 and BF16 backward kernels share the half-precision module. FP32
    // stays in the base module.
    let bwd_module = match dtype {
        DType::F16 | DType::BF16 => VARLEN_ATTENTION_BWD_HALF_MODULE,
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

    let (total_tokens_q, total_tokens_k) = (totals.q, totals.k);

    // dq uses the Q head layout (num_heads) and the input dtype. One block
    // owns each Q row, so the kernel stores it once with no atomics.
    // dk/dv use the KV head layout (num_kv_heads). Under GQA several Q heads
    // and every Q block add into one KV row, so both kernels accumulate dk/dv
    // with float atomicAdd into zeroed F32 buffers. The F16 and BF16 paths
    // cast them to the input dtype after the launch.
    let dq = Tensor::<CudaRuntime>::zeros(&[total_tokens_q, num_heads, head_dim], dtype, device)?;
    let kv_grad_shape = [total_tokens_k, num_kv_heads, head_dim];
    let dk_acc = Tensor::<CudaRuntime>::zeros(&kv_grad_shape, DType::F32, device)?;
    let dv_acc = Tensor::<CudaRuntime>::zeros(&kv_grad_shape, DType::F32, device)?;

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

    let blocks = num_q_blocks
        .checked_mul(num_heads)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::InvalidArgument {
            arg: "max_seqlen_q",
            reason: format!(
                "varlen attention bwd: {num_q_blocks} query blocks of {num_heads} heads pass \
                 the grid limit of {} blocks. Split the batch",
                u32::MAX
            ),
        })?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
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
    let dk_ptr = dk_acc.ptr();
    let dv_ptr = dv_acc.ptr();
    let scale = (head_dim as f32).sqrt().recip();
    let batch_i32 = batch_size as i32;
    let nh_i32 = num_heads as i32;
    let nkv_i32 = num_kv_heads as i32;
    let msq_i32 = max_seqlen_q as i32;
    let msk_i32 = max_seqlen_k as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };
    // Same value the forward passes, so both mask the same keys.
    let window_i32 = window_size_arg(window_size);

    // SAFETY: the nineteen args match `varlen_flash_attention_bwd_{HD}_{fp32,fp16,bf16}{tile}`
    // in order and type: device pointers to Q, K, V, O in `dtype` (which picks the
    // kernel), L in F32, dO in `dtype`, the two I32 offset tensors, dQ in `dtype`, dK and
    // dV in F32, then `batch_size`, `num_heads`, `num_kv_heads`, `max_seqlen_q`,
    // `max_seqlen_k` as `int`, `scale` as `float`, `causal` and `window_size` as `int`.
    // `check_bwd_args` holds every input to its contract shape and dtype and every count
    // to `i32`, and the copies above make them contiguous: Q, O and dO hold
    // `Tq * num_heads * head_dim` values, K and V `Tk * num_kv_heads * head_dim`, L
    // `Tq * num_heads`, each offset tensor `batch_size + 1`. dQ, dK and dV are fresh
    // zeroed buffers of the Q and KV shapes. The trait wiring checks the host offsets
    // against `Tq` and `Tk`. The device offsets must hold the same values, which is the
    // caller's contract. Every buffer outlives the launch: numr frees and reuses device
    // memory in order on this stream, so a later free waits for the kernel.
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
        builder.arg(&window_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!(
                "varlen attention bwd: launch of `{kernel_name}` failed ({e:?}); \
                 check the CUDA device state and the kernel module build"
            ),
        })?;
    }

    // No stream sync here. Every atomicAdd completes inside the kernel. The
    // cast below, every later boostr/numr kernel and every device-to-host
    // copy run on this same compute stream, so stream order covers each
    // reader. A host sync is also illegal inside CUDA graph capture.
    if dtype == DType::F32 {
        return Ok((dq, dk_acc, dv_acc));
    }
    let dk = client.cast(&dk_acc, dtype)?;
    let dv = client.cast(&dv_acc, dtype)?;

    Ok((dq, dk, dv))
}
