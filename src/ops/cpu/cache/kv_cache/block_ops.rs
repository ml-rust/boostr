//! `copy_blocks` / `swap_blocks` CPU reference implementation.
//!
//! Split out of `kv_cache.rs` to keep concerns separated (that file is
//! already at the `cpu/*.rs` line-count ceiling). Plain nested loops with
//! per-dtype byte-size copies — clarity over speed, since this is the
//! cross-backend parity reference, not a hot path.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Shared validation for `copy_blocks` and `swap_blocks`.
///
/// Returns `(num_pairs, block_mapping_host)` on success.
#[allow(clippy::too_many_arguments)]
fn validate_block_mapping(
    cache_a: &Tensor<CpuRuntime>,
    cache_b: &Tensor<CpuRuntime>,
    arg_a: &'static str,
    arg_b: &'static str,
    block_mapping: &Tensor<CpuRuntime>,
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
    let mapping = block_mapping.to_vec::<i32>();
    if !mapping.len().is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "block_mapping",
            reason: format!("numel {} must be even (src/dst pairs)", mapping.len()),
        });
    }
    let num_pairs = mapping.len() / 2;

    let num_blocks_a = cache_a.shape()[0];
    let num_blocks_b = cache_b.shape()[0];
    for (i, &b) in mapping.iter().enumerate() {
        if b < 0 || (b as usize) >= num_blocks_a || (b as usize) >= num_blocks_b {
            return Err(Error::InvalidArgument {
                arg: "block_mapping",
                reason: format!(
                    "block index {b} at position {i} out of range ({arg_a} has {num_blocks_a} blocks, {arg_b} has {num_blocks_b} blocks)"
                ),
            });
        }
    }

    Ok((num_pairs, mapping))
}

pub(super) fn copy_blocks(
    key_cache: &Tensor<CpuRuntime>,
    value_cache: &Tensor<CpuRuntime>,
    block_mapping: &Tensor<CpuRuntime>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let (num_pairs, mapping) = validate_block_mapping(
        key_cache,
        value_cache,
        "key_cache",
        "value_cache",
        block_mapping,
        num_heads,
        head_dim,
        block_size,
    )?;

    let elem_size = key_cache.dtype().size_in_bytes();
    let row_bytes = head_dim * elem_size;
    let head_stride_bytes = row_bytes;
    let slot_stride_bytes = num_heads * head_stride_bytes;
    let block_stride_bytes = block_size * slot_stride_bytes;

    let kc_ptr = key_cache.ptr() as *mut u8;
    let vc_ptr = value_cache.ptr() as *mut u8;

    for p in 0..num_pairs {
        let src_block = mapping[p * 2] as usize;
        let dst_block = mapping[p * 2 + 1] as usize;
        for slot in 0..block_size {
            for h in 0..num_heads {
                let src_off = src_block * block_stride_bytes
                    + slot * slot_stride_bytes
                    + h * head_stride_bytes;
                let dst_off = dst_block * block_stride_bytes
                    + slot * slot_stride_bytes
                    + h * head_stride_bytes;
                // `copy` (not `copy_nonoverlapping`): src_block == dst_block
                // is a valid no-op pair, and the two ranges then alias.
                unsafe {
                    std::ptr::copy(kc_ptr.add(src_off), kc_ptr.add(dst_off), row_bytes);
                    std::ptr::copy(vc_ptr.add(src_off), vc_ptr.add(dst_off), row_bytes);
                }
            }
        }
    }

    Ok(())
}

pub(super) fn swap_blocks(
    src_cache: &Tensor<CpuRuntime>,
    dst_cache: &Tensor<CpuRuntime>,
    block_mapping: &Tensor<CpuRuntime>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let (num_pairs, mapping) = validate_block_mapping(
        src_cache,
        dst_cache,
        "src_cache",
        "dst_cache",
        block_mapping,
        num_heads,
        head_dim,
        block_size,
    )?;

    let elem_size = src_cache.dtype().size_in_bytes();
    let row_bytes = head_dim * elem_size;
    let head_stride_bytes = row_bytes;
    let slot_stride_bytes = num_heads * head_stride_bytes;
    let block_stride_bytes = block_size * slot_stride_bytes;

    let src_ptr = src_cache.ptr() as *const u8;
    let dst_ptr = dst_cache.ptr() as *mut u8;

    for p in 0..num_pairs {
        let src_block = mapping[p * 2] as usize;
        let dst_block = mapping[p * 2 + 1] as usize;
        for slot in 0..block_size {
            for h in 0..num_heads {
                let src_off = src_block * block_stride_bytes
                    + slot * slot_stride_bytes
                    + h * head_stride_bytes;
                let dst_off = dst_block * block_stride_bytes
                    + slot * slot_stride_bytes
                    + h * head_stride_bytes;
                // src_cache and dst_cache are two distinct buffers (never the
                // same tensor), so the ranges never alias here.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(src_off),
                        dst_ptr.add(dst_off),
                        row_bytes,
                    );
                }
            }
        }
    }

    Ok(())
}
