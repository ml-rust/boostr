//! `copy_blocks` / `swap_blocks` CUDA dispatch.
//!
//! Split out of `kv_cache.rs` to stay under the `cuda/*.rs` 400-line limit.
//! Both kernels live in the same `reshape_and_cache` PTX unit as
//! `reshape_and_cache_*` and share its module cache entry.

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{self, RESHAPE_AND_CACHE_MODULE};

/// Shared validation for `copy_blocks` and `swap_blocks`.
///
/// Shape and dtype checks read tensor metadata only. The block-index bounds
/// check reads `block_mapping` to host; see the comment at that call.
///
/// Returns `(num_pairs, block_mapping_host)`.
#[allow(clippy::too_many_arguments)]
fn validate_block_mapping(
    cache_a: &Tensor<CudaRuntime>,
    cache_b: &Tensor<CudaRuntime>,
    arg_a: &'static str,
    arg_b: &'static str,
    block_mapping: &Tensor<CudaRuntime>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(usize, Vec<i32>)> {
    if num_heads == 0 || head_dim == 0 || block_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "block_size",
            reason: format!(
                "num_heads ({num_heads}), head_dim ({head_dim}) and block_size ({block_size}) must all be non-zero"
            ),
        });
    }

    for (arg, cache) in [(arg_a, cache_a), (arg_b, cache_b)] {
        let shape = cache.shape();
        if shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "expected 4D [num_blocks, block_size, num_heads, head_dim], got {}D",
                    shape.len()
                ),
            });
        }
        if shape[1] != block_size || shape[2] != num_heads || shape[3] != head_dim {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "shape [.., {}, {}, {}] does not match block_size={}, num_heads={}, head_dim={}",
                    shape[1], shape[2], shape[3], block_size, num_heads, head_dim
                ),
            });
        }
    }

    if cache_b.dtype() != cache_a.dtype() {
        return Err(Error::InvalidArgument {
            arg: arg_b,
            reason: format!(
                "{} dtype {:?} does not match {} dtype {:?}",
                arg_b,
                cache_b.dtype(),
                arg_a,
                cache_a.dtype()
            ),
        });
    }

    // Checked here, not at the kernel-name match, so an unsupported dtype is
    // rejected even when the mapping is empty and the op returns early. The
    // CPU backend rejects the same inputs.
    match cache_a.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => {}
        dt => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {dt:?}"),
            });
        }
    }

    if block_mapping.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg: "block_mapping",
            reason: format!("expected I32, got {:?}", block_mapping.dtype()),
        });
    }
    let mapping_len = block_mapping.numel();
    if !mapping_len.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "block_mapping",
            reason: format!("numel {mapping_len} must be even (src/dst pairs)"),
        });
    }
    let num_pairs = mapping_len / 2;

    // Reads indices only, never K/V, and block copy/swap runs once per
    // scheduling step, not per token. Launching unchecked corrupts device
    // memory instead of returning an error.
    let mapping_host = block_mapping.to_vec::<i32>();
    let num_blocks_a = cache_a.shape()[0];
    let num_blocks_b = cache_b.shape()[0];
    for (i, &b) in mapping_host.iter().enumerate() {
        if b < 0 || (b as usize) >= num_blocks_a || (b as usize) >= num_blocks_b {
            return Err(Error::InvalidArgument {
                arg: "block_mapping",
                reason: format!(
                    "block index {b} at position {i} out of range ({arg_a} has {num_blocks_a} blocks, {arg_b} has {num_blocks_b} blocks)"
                ),
            });
        }
    }

    Ok((num_pairs, mapping_host))
}

fn vec_size_for(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        _ => 8,
    }
}

pub(super) fn copy_blocks(
    client: &CudaClient,
    key_cache: &Tensor<CudaRuntime>,
    value_cache: &Tensor<CudaRuntime>,
    block_mapping: &Tensor<CudaRuntime>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let (num_pairs, _mapping) = validate_block_mapping(
        key_cache,
        value_cache,
        "key_cache",
        "value_cache",
        block_mapping,
        num_heads,
        head_dim,
        block_size,
    )?;
    if num_pairs == 0 {
        return Ok(());
    }

    let dtype = key_cache.dtype();
    let kernel_name = match dtype {
        DType::F32 => "copy_blocks_f32",
        DType::F16 => "copy_blocks_f16",
        DType::BF16 => "copy_blocks_bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {dtype:?} for copy_blocks"),
            });
        }
    };

    let threads = head_dim.div_ceil(vec_size_for(dtype)).max(1);

    let device = key_cache.device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, RESHAPE_AND_CACHE_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let cfg = LaunchConfig {
        grid_dim: (num_pairs as u32, num_heads as u32, block_size as u32),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let kc_ptr = key_cache.ptr();
    let vc_ptr = value_cache.ptr();
    let bm_ptr = block_mapping.ptr();
    let np_i32 = num_pairs as i32;
    let nh_i32 = num_heads as i32;
    let hd_i32 = head_dim as i32;
    let bs_i32 = block_size as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&kc_ptr);
        builder.arg(&vc_ptr);
        builder.arg(&bm_ptr);
        builder.arg(&np_i32);
        builder.arg(&nh_i32);
        builder.arg(&hd_i32);
        builder.arg(&bs_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("copy_blocks kernel launch failed: {e:?}"),
        })?;
    }

    Ok(())
}

pub(super) fn swap_blocks(
    client: &CudaClient,
    src_cache: &Tensor<CudaRuntime>,
    dst_cache: &Tensor<CudaRuntime>,
    block_mapping: &Tensor<CudaRuntime>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let (num_pairs, _mapping) = validate_block_mapping(
        src_cache,
        dst_cache,
        "src_cache",
        "dst_cache",
        block_mapping,
        num_heads,
        head_dim,
        block_size,
    )?;
    if num_pairs == 0 {
        return Ok(());
    }

    let dtype = src_cache.dtype();
    let kernel_name = match dtype {
        DType::F32 => "swap_blocks_f32",
        DType::F16 => "swap_blocks_f16",
        DType::BF16 => "swap_blocks_bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {dtype:?} for swap_blocks"),
            });
        }
    };

    let threads = head_dim.div_ceil(vec_size_for(dtype)).max(1);

    let device = src_cache.device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, RESHAPE_AND_CACHE_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let cfg = LaunchConfig {
        grid_dim: (num_pairs as u32, num_heads as u32, block_size as u32),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let src_ptr = src_cache.ptr();
    let dst_ptr = dst_cache.ptr();
    let bm_ptr = block_mapping.ptr();
    let np_i32 = num_pairs as i32;
    let nh_i32 = num_heads as i32;
    let hd_i32 = head_dim as i32;
    let bs_i32 = block_size as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&bm_ptr);
        builder.arg(&np_i32);
        builder.arg(&nh_i32);
        builder.arg(&hd_i32);
        builder.arg(&bs_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("swap_blocks kernel launch failed: {e:?}"),
        })?;
    }

    Ok(())
}
