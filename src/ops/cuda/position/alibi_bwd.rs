//! Backward pass for the materialized biased-attention path (F32/F16/BF16).
//!
//! Launches the `alibi_bwd` kernel module: one softmax-Jacobian kernel followed
//! by one kernel each for dQ, dK and dV. `grad_probs = grad_output @ V^T` is a
//! plain numr matmul — there is no kernel for it here.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, ALIBI_BWD_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaFunction, LaunchConfig};
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Threads per block for the softmax-Jacobian row reduction.
///
/// MUST be a power of two: the kernel's tree reduction halves `blockDim.x` each
/// step and drops lanes otherwise.
const SOFTMAX_BLOCK: u32 = 256;

/// Maximum threads per block for the head-dim kernels.
const HEAD_DIM_BLOCK_MAX: u32 = 256;

/// Shapes shared by every launch below.
struct Dims {
    batch_size: i32,
    num_heads: i32,
    seq_len_q: i32,
    seq_len_k: i32,
    head_dim: i32,
}

fn check_shape(t: &Tensor<CudaRuntime>, arg: &'static str, expected: &[usize]) -> Result<()> {
    if t.shape() != expected {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected shape {:?}, got {:?}", expected, t.shape()),
        });
    }
    if !t.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg,
            reason: "alibi_attention_bwd requires contiguous inputs".into(),
        });
    }
    Ok(())
}

/// Launch `alibi_backward_grad_q_*` or `alibi_backward_grad_k_*`.
///
/// Both take `(scores, other, out, scale, b, nh, sq, sk, hd)` and index
/// `blockIdx.z` = batch*heads + head, `blockIdx.y` = row, `blockIdx.x *
/// blockDim.x + threadIdx.x` = head-dim element. `rows` is `seq_len_q` for dQ
/// and `seq_len_k` for dK.
#[allow(clippy::too_many_arguments)]
fn launch_scaled(
    client: &CudaClient,
    func: &CudaFunction,
    grad_scores: &Tensor<CudaRuntime>,
    other: &Tensor<CudaRuntime>,
    out: &Tensor<CudaRuntime>,
    scale: f32,
    d: &Dims,
    rows: usize,
    label: &str,
) -> Result<()> {
    let block_x = (d.head_dim as u32).clamp(1, HEAD_DIM_BLOCK_MAX);
    let cfg = LaunchConfig {
        grid_dim: (
            (d.head_dim as u32).div_ceil(block_x),
            rows as u32,
            (d.batch_size * d.num_heads) as u32,
        ),
        block_dim: (block_x, 1, 1),
        shared_mem_bytes: 0,
    };

    let gs_ptr = grad_scores.ptr();
    let other_ptr = other.ptr();
    let out_ptr = out.ptr();

    unsafe {
        let mut builder = client.stream().launch_builder(func);
        builder.arg(&gs_ptr);
        builder.arg(&other_ptr);
        builder.arg(&out_ptr);
        builder.arg(&scale);
        builder.arg(&d.batch_size);
        builder.arg(&d.num_heads);
        builder.arg(&d.seq_len_q);
        builder.arg(&d.seq_len_k);
        builder.arg(&d.head_dim);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("ALiBi backward {label} kernel failed: {e:?}"),
        })?;
    }

    Ok(())
}

/// Backward for the materialized biased-attention path. Returns (dQ, dK, dV).
#[allow(clippy::too_many_arguments)]
pub(super) fn alibi_attention_bwd_impl(
    client: &CudaClient,
    grad_output: &Tensor<CudaRuntime>,
    probs: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let probs_shape = probs.shape();
    if probs_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "probs",
            reason: format!("expected 4D [B, H, S_q, S_k], got {}D", probs_shape.len()),
        });
    }
    let seq_len_q = probs_shape[2];
    let seq_len_k = probs_shape[3];

    check_shape(
        probs,
        "probs",
        &[batch_size, num_heads, seq_len_q, seq_len_k],
    )?;
    check_shape(
        grad_output,
        "grad_output",
        &[batch_size, num_heads, seq_len_q, head_dim],
    )?;
    check_shape(q, "q", &[batch_size, num_heads, seq_len_q, head_dim])?;
    check_shape(k, "k", &[batch_size, num_heads, seq_len_k, head_dim])?;
    check_shape(v, "v", &[batch_size, num_heads, seq_len_k, head_dim])?;

    let dtype = q.dtype();
    for t in [grad_output, probs, k, v] {
        if t.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                got: t.dtype(),
            });
        }
    }

    // The kernels are named with `_f32` / `_f16` / `_bf16` suffixes.
    let dtype_suffix = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {dtype:?} for alibi_attention_bwd"),
            });
        }
    };

    let device = q.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, ALIBI_BWD_MODULE)?;

    let d = Dims {
        batch_size: batch_size as i32,
        num_heads: num_heads as i32,
        seq_len_q: seq_len_q as i32,
        seq_len_k: seq_len_k as i32,
        head_dim: head_dim as i32,
    };

    // Step 1: grad_probs = grad_output @ V^T  ->  [B, H, S_q, S_k].
    // numr matmul, not a kernel of ours.
    let v_t = v.transpose(-2, -1)?.contiguous()?;
    let grad_probs = client.matmul(grad_output, &v_t)?;

    let grad_scores = Tensor::<CudaRuntime>::empty(
        &[batch_size, num_heads, seq_len_q, seq_len_k],
        dtype,
        device,
    )?;
    let grad_q =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device)?;
    let grad_k =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_k, head_dim], dtype, device)?;
    let grad_v =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_k, head_dim], dtype, device)?;

    // Step 2: softmax Jacobian.
    // grid = (1, S_q, B*H), block = (SOFTMAX_BLOCK, 1, 1), and the kernel's
    // `extern __shared__ float sdata[]` needs blockDim.x floats.
    {
        let name = format!("alibi_softmax_backward_{dtype_suffix}");
        let func = kernels::get_kernel_function(&module, &name)?;

        let cfg = LaunchConfig {
            grid_dim: (1, seq_len_q as u32, (batch_size * num_heads) as u32),
            block_dim: (SOFTMAX_BLOCK, 1, 1),
            shared_mem_bytes: SOFTMAX_BLOCK * std::mem::size_of::<f32>() as u32,
        };

        let gp_ptr = grad_probs.ptr();
        let p_ptr = probs.ptr();
        let gs_ptr = grad_scores.ptr();

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&gp_ptr);
            builder.arg(&p_ptr);
            builder.arg(&gs_ptr);
            builder.arg(&d.batch_size);
            builder.arg(&d.num_heads);
            builder.arg(&d.seq_len_q);
            builder.arg(&d.seq_len_k);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("ALiBi backward softmax kernel failed: {e:?}"),
            })?;
        }
    }

    // Step 3: dQ = (grad_scores @ K) * scale, one row per query position.
    {
        let name = format!("alibi_backward_grad_q_{dtype_suffix}");
        let func = kernels::get_kernel_function(&module, &name)?;
        launch_scaled(
            client,
            &func,
            &grad_scores,
            k,
            &grad_q,
            scale,
            &d,
            seq_len_q,
            "grad_q",
        )?;
    }

    // Step 4: dK = (grad_scores^T @ Q) * scale, one row per key position.
    {
        let name = format!("alibi_backward_grad_k_{dtype_suffix}");
        let func = kernels::get_kernel_function(&module, &name)?;
        launch_scaled(
            client,
            &func,
            &grad_scores,
            q,
            &grad_k,
            scale,
            &d,
            seq_len_k,
            "grad_k",
        )?;
    }

    // Step 5: dV = probs^T @ grad_output. No scale — the bias path applies
    // `scale` before the softmax, so it reaches dV through `probs` alone.
    {
        let name = format!("alibi_backward_grad_v_{dtype_suffix}");
        let func = kernels::get_kernel_function(&module, &name)?;

        let block_x = (head_dim as u32).clamp(1, HEAD_DIM_BLOCK_MAX);
        let cfg = LaunchConfig {
            grid_dim: (
                (head_dim as u32).div_ceil(block_x),
                seq_len_k as u32,
                (batch_size * num_heads) as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };

        let p_ptr = probs.ptr();
        let go_ptr = grad_output.ptr();
        let gv_ptr = grad_v.ptr();

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&p_ptr);
            builder.arg(&go_ptr);
            builder.arg(&gv_ptr);
            builder.arg(&d.batch_size);
            builder.arg(&d.num_heads);
            builder.arg(&d.seq_len_q);
            builder.arg(&d.seq_len_k);
            builder.arg(&d.head_dim);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("ALiBi backward grad_v kernel failed: {e:?}"),
            })?;
        }
    }

    Ok((grad_q, grad_k, grad_v))
}
