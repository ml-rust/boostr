//! CUDA kernel launchers for the fused-QKV epilogue.
//!
//! `fused_qkv_bias_split` fuses bias-add + split + `transpose(1,2)` into one
//! pass. `fused_output_bias_residual` fuses bias-add + residual-add.
//! `fused_qkv_concat` is the backward gather, the exact inverse of the split.
//! All three exist only for F32/F64 — see [`fused_qkv_dtype_suffix`]; every
//! other dtype must fall back to the generic elementwise path in the caller.
//! "No bias" is signalled by a null device pointer; the kernels check
//! `bias != nullptr` before dereferencing it.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FUSED_QKV_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// `f32`/`f64` kernel suffix, or `None` when no fused kernel exists for
/// `dtype` (the caller must fall back to the generic path).
pub(super) fn fused_qkv_dtype_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("f32"),
        DType::F64 => Some("f64"),
        _ => None,
    }
}

fn launch_cfg(total: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = (total as u32).div_ceil(threads);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch `fused_qkv_bias_split_{suffix}`. `qkv` is `[B*S, total_proj]`,
/// laid out `[Hq | Hkv | Hkv]`. Returns `(Q, K, V)` already in
/// `[B, heads, S, D]` layout — the split and the `transpose(1,2)` the
/// generic path does separately are fused into the same write.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_bias_split(
    client: &CudaClient,
    qkv: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    suffix: &str,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    total_proj: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let dtype = qkv.dtype();
    let device = qkv.device().clone();

    let q =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len, head_dim], dtype, &device)?;
    let k = Tensor::<CudaRuntime>::empty(
        &[batch_size, num_kv_heads, seq_len, head_dim],
        dtype,
        &device,
    )?;
    let v = Tensor::<CudaRuntime>::empty(
        &[batch_size, num_kv_heads, seq_len, head_dim],
        dtype,
        &device,
    )?;

    let module = kernels::get_or_load_module(client.context(), device.id(), FUSED_QKV_MODULE)?;
    let func = kernels::get_kernel_function(&module, &format!("fused_qkv_bias_split_{suffix}"))?;
    let cfg = launch_cfg(batch_size * seq_len * total_proj);

    let qkv_ptr = qkv.ptr();
    // No bias -> null pointer (see module doc).
    let bias_ptr: u64 = bias.map(|b| b.ptr()).unwrap_or(0);
    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let b_u = batch_size as u32;
    let s_u = seq_len as u32;
    let nh_u = num_heads as u32;
    let nkv_u = num_kv_heads as u32;
    let hd_u = head_dim as u32;
    let tp_u = total_proj as u32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&qkv_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&b_u);
        builder.arg(&s_u);
        builder.arg(&nh_u);
        builder.arg(&nkv_u);
        builder.arg(&hd_u);
        builder.arg(&tp_u);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_qkv_bias_split launch failed: {:?}", e),
        })?;
    }

    Ok((q, k, v))
}

/// Launch `fused_output_bias_residual_{suffix}` over flat `[B*S, H]` tensors.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_output_bias_residual(
    client: &CudaClient,
    proj: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    residual: &Tensor<CudaRuntime>,
    suffix: &str,
    total: usize,
    hidden_dim: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = proj.dtype();
    let device = proj.device().clone();
    let output = Tensor::<CudaRuntime>::empty(proj.shape(), dtype, &device)?;

    let module = kernels::get_or_load_module(client.context(), device.id(), FUSED_QKV_MODULE)?;
    let func =
        kernels::get_kernel_function(&module, &format!("fused_output_bias_residual_{suffix}"))?;
    let cfg = launch_cfg(total);

    let proj_ptr = proj.ptr();
    // No bias -> null pointer (see module doc).
    let bias_ptr: u64 = bias.map(|b| b.ptr()).unwrap_or(0);
    let residual_ptr = residual.ptr();
    let out_ptr = output.ptr();
    let total_u = total as u32;
    let hidden_u = hidden_dim as u32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&proj_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&residual_ptr);
        builder.arg(&out_ptr);
        builder.arg(&total_u);
        builder.arg(&hidden_u);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_output_bias_residual launch failed: {:?}", e),
        })?;
    }

    Ok(output)
}

/// Launch `fused_qkv_concat_{suffix}`: the exact inverse of
/// `fused_qkv_bias_split`, gathering `dq`/`dk`/`dv` `[B, heads, S, D]` into
/// `d_qkv` `[B*S, total_proj]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_qkv_concat(
    client: &CudaClient,
    dq: &Tensor<CudaRuntime>,
    dk: &Tensor<CudaRuntime>,
    dv: &Tensor<CudaRuntime>,
    suffix: &str,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    total_proj: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = dq.dtype();
    let device = dq.device().clone();
    let d_qkv = Tensor::<CudaRuntime>::empty(&[batch_size * seq_len, total_proj], dtype, &device)?;

    let module = kernels::get_or_load_module(client.context(), device.id(), FUSED_QKV_MODULE)?;
    let func = kernels::get_kernel_function(&module, &format!("fused_qkv_concat_{suffix}"))?;
    let cfg = launch_cfg(batch_size * seq_len * total_proj);

    let dq_ptr = dq.ptr();
    let dk_ptr = dk.ptr();
    let dv_ptr = dv.ptr();
    let out_ptr = d_qkv.ptr();
    let b_u = batch_size as u32;
    let s_u = seq_len as u32;
    let nh_u = num_heads as u32;
    let nkv_u = num_kv_heads as u32;
    let hd_u = head_dim as u32;
    let tp_u = total_proj as u32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&dq_ptr);
        builder.arg(&dk_ptr);
        builder.arg(&dv_ptr);
        builder.arg(&out_ptr);
        builder.arg(&b_u);
        builder.arg(&s_u);
        builder.arg(&nh_u);
        builder.arg(&nkv_u);
        builder.arg(&hd_u);
        builder.arg(&tp_u);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_qkv_concat launch failed: {:?}", e),
        })?;
    }

    Ok(d_qkv)
}
