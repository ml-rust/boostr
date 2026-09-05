//! Device shared-memory helpers shared by every Flash Attention CUDA launcher:
//! the device's opt-in limit, the forward/backward shared-memory formulas, and
//! the opt-in attribute call.
//!
//! Split out of `flash_utils.rs`; block-config selection lives in
//! `flash_block_config.rs` and parameter validation in `flash_params.rs`.
//! `flash_utils` re-exports all three, so existing import paths still resolve.

use crate::error::{Error, Result};
use cudarc::driver::safe::CudaFunction;
use cudarc::driver::sys;

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

/// Shared memory bytes the flash-attention BACKWARD kernel needs.
///
/// Layout in `flash_v2_bwd.cu` (`flash_attention_bwd_*_impl`):
/// `[K: BLOCK_N x HEAD_DIM][V: BLOCK_N x HEAD_DIM][Q: BLOCK_M x HEAD_DIM][dO: BLOCK_M x HEAD_DIM]`,
/// stored in the kernel's element type with NO `+1` bank-conflict padding — unlike
/// the forward layout in `compute_smem`.
pub(super) fn compute_bwd_smem(
    block_m: usize,
    block_n: usize,
    head_dim: usize,
    elem_bytes: usize,
) -> usize {
    (2 * block_n + 2 * block_m) * head_dim * elem_bytes
}

/// Set dynamic shared memory attribute if it reaches the 48KB default cap.
///
/// Requests of exactly 48KB go through the opt-in path too: some backward block
/// configs land on that boundary, and raising the cap explicitly is always valid.
pub(crate) fn set_smem_attribute(func: &CudaFunction, smem_size: usize) -> Result<()> {
    if smem_size < 48 * 1024 {
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

    // cudarc's own accessor. This previously read `CudaFunction`'s private
    // `cu_function` field by pointer offset; that struct is `repr(Rust)`, so
    // field order is not guaranteed and a layout change would have silently
    // yielded a handle to some other kernel rather than failing to compile.
    func.set_attribute(
        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_size as i32,
    )
    .map_err(|e| Error::KernelError {
        reason: format!(
            "failed to set dynamic shared memory to {}KB: {:?}",
            smem_size / 1024,
            e
        ),
    })?;

    Ok(())
}
