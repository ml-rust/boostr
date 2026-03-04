//! Shared utilities for Flash Attention v2: parameter validation, block config,
//! shared memory helpers.

use crate::error::{Error, Result};
use cudarc::driver::safe::CudaFunction;
use cudarc::driver::sys;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Validated attention parameters extracted from tensor shapes.
pub(super) struct AttentionParams {
    pub batch_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub seq_len_q: usize,
    pub seq_len_k: usize,
    pub head_dim: usize,
    pub block_m: usize,
    pub block_n: usize,
    /// Whether to use the small-memory kernel variant (_sm suffix)
    pub use_sm_kernel: bool,
}

/// Validate Q/K/V shapes and extract parameters.
pub(super) fn validate_qkv(
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<AttentionParams> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    if k_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("expected 4D, got {}D", k_shape.len()),
        });
    }
    if v_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("expected 4D, got {}D", v_shape.len()),
        });
    }
    if q_shape[1] != num_heads {
        return Err(Error::InvalidArgument {
            arg: "num_heads",
            reason: format!("num_heads={} but q dim 1 is {}", num_heads, q_shape[1]),
        });
    }
    if k_shape[1] != num_kv_heads {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_kv_heads={} but k dim 1 is {}",
                num_kv_heads, k_shape[1]
            ),
        });
    }
    if q_shape[3] != head_dim || k_shape[3] != head_dim || v_shape[3] != head_dim {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "head_dim={} but q.D={}, k.D={}, v.D={}",
                head_dim, q_shape[3], k_shape[3], v_shape[3]
            ),
        });
    }
    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(Error::InvalidArgument {
            arg: "batch_size",
            reason: format!(
                "batch mismatch: q.B={}, k.B={}, v.B={}",
                q_shape[0], k_shape[0], v_shape[0]
            ),
        });
    }
    if k_shape[2] != v_shape[2] {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("k seq_len={} != v seq_len={}", k_shape[2], v_shape[2]),
        });
    }
    if num_heads % num_kv_heads != 0 {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            ),
        });
    }

    let dtype = q.dtype();
    if k.dtype() != dtype || v.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
                dtype,
                k.dtype(),
                v.dtype()
            ),
        });
    }
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg: "contiguity",
            reason: "Flash Attention requires contiguous Q, K, V tensors".into(),
        });
    }

    let elem_bytes = q.dtype().size_in_bytes();
    let (block_m, block_n, use_sm_kernel) = block_config(head_dim, elem_bytes)?;

    Ok(AttentionParams {
        batch_size: q_shape[0],
        num_heads,
        num_kv_heads,
        seq_len_q: q_shape[2],
        seq_len_k: k_shape[2],
        head_dim,
        block_m,
        block_n,
        use_sm_kernel,
    })
}

/// Query the device's max dynamic shared memory per block (opt-in).
pub(super) fn device_max_smem() -> usize {
    unsafe {
        let mut cuda_dev: i32 = 0;
        sys::cuCtxGetDevice(&mut cuda_dev);
        let mut max_smem: i32 = 0;
        sys::cuDeviceGetAttribute(
            &mut max_smem,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            cuda_dev,
        );
        max_smem as usize
    }
}

/// Compute shared memory bytes for given block config, head_dim, and element size.
pub(super) fn compute_smem(
    block_m: usize,
    block_n: usize,
    head_dim: usize,
    elem_bytes: usize,
) -> usize {
    let head_stride = head_dim + 1; // +1 padding for bank conflict avoidance
    (block_m * head_stride + 2 * block_n * head_stride) * elem_bytes
}

/// Standard (large) block config — best performance on high-smem GPUs.
fn block_config_large(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        32 => Some((128, 128)),
        64 => Some((128, 128)),
        96 => Some((64, 128)),
        128 => Some((128, 64)),
        192 => Some((64, 64)),
        256 => Some((64, 64)),
        _ => None,
    }
}

/// Small-memory block config — works on GPUs with <=100KB shared memory.
/// These have corresponding `_sm` kernel variants in flash_v2.cu.
fn block_config_small(head_dim: usize) -> Option<(usize, usize)> {
    match head_dim {
        96 => Some((32, 32)),
        128 => Some((64, 32)),
        192 => Some((32, 16)),
        256 => Some((16, 16)),
        _ => None,
    }
}

/// Get block configuration for a head dimension, accounting for device shared memory limits.
/// Returns (block_m, block_n, use_sm_kernel).
pub(super) fn block_config(head_dim: usize, elem_bytes: usize) -> Result<(usize, usize, bool)> {
    // Try large config first
    if let Some((bm, bn)) = block_config_large(head_dim) {
        let smem = compute_smem(bm, bn, head_dim, elem_bytes);
        if smem <= device_max_smem() {
            return Ok((bm, bn, false));
        }
    }

    // Fall back to small-memory config
    if let Some((bm, bn)) = block_config_small(head_dim) {
        let smem = compute_smem(bm, bn, head_dim, elem_bytes);
        if smem <= device_max_smem() {
            return Ok((bm, bn, true));
        }
    }

    Err(Error::InvalidArgument {
        arg: "head_dim",
        reason: format!(
            "unsupported head_dim={} for this GPU (max shared memory: {}KB). Supported: 32, 64, 96, 128, 192, 256",
            head_dim,
            device_max_smem() / 1024
        ),
    })
}

/// Set dynamic shared memory attribute if >48KB.
pub(crate) fn set_smem_attribute(func: &CudaFunction, smem_size: usize) -> Result<()> {
    if smem_size <= 48 * 1024 {
        return Ok(());
    }

    let max_shared_mem = unsafe {
        let mut cuda_dev: i32 = 0;
        sys::cuCtxGetDevice(&mut cuda_dev);
        let mut max_smem: i32 = 0;
        sys::cuDeviceGetAttribute(
            &mut max_smem,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            cuda_dev,
        );
        max_smem as usize
    };

    if smem_size > max_shared_mem {
        return Err(Error::KernelError {
            reason: format!(
                "shared memory {}KB exceeds device limit {}KB",
                smem_size / 1024,
                max_shared_mem / 1024
            ),
        });
    }

    // Extract CUfunction handle (second field of CudaFunction)
    let cu_function: sys::CUfunction = unsafe {
        let kernel_ptr = func as *const _ as *const usize;
        std::ptr::read(kernel_ptr.add(1)) as sys::CUfunction
    };

    unsafe {
        let result = sys::cuFuncSetAttribute(
            cu_function,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_size as i32,
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::KernelError {
                reason: format!(
                    "failed to set dynamic shared memory to {}KB: {:?}",
                    smem_size / 1024,
                    result
                ),
            });
        }
    }

    Ok(())
}
