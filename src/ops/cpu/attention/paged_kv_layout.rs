//! Paged KV cache layout mapping for the CPU backend.
//!
//! The cache is `[num_blocks, block_size, num_kv_heads, head_dim]` and the
//! block table maps a sequence's logical block index to a physical one, per the
//! layout contract on `PagedAttentionOps`. These two functions are the only
//! place that mapping is written down on CPU; the attention itself runs on the
//! gathered contiguous tensors.

use crate::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Element offset of token `(physical_block, block_offset, kv_head)` in a cache
/// laid out `[num_blocks, block_size, num_kv_heads, head_dim]`.
#[inline]
fn cache_offset(
    physical_block: usize,
    block_offset: usize,
    kv_head: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> usize {
    ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim
}

/// Gather paged KV blocks into a contiguous `[B, num_kv_heads, seq_len_k, head_dim]`
/// tensor.
///
/// The caller expands the KV heads to query heads afterwards; keeping them
/// separate here is what makes grouped-query caches gather correctly.
pub(super) fn gather_paged_kv(
    kv_blocks: &Tensor<CpuRuntime>,
    block_table: &Tensor<CpuRuntime>,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor<CpuRuntime>> {
    let kv_data = kv_blocks.to_vec::<f32>();
    let bt_data = block_table.to_vec::<i32>();
    let max_num_blocks = block_table.shape()[1];

    let mut out = vec![0.0f32; batch_size * num_kv_heads * seq_len_k * head_dim];

    for b in 0..batch_size {
        for t in 0..seq_len_k {
            let logical_block = t / block_size;
            let block_offset = t % block_size;
            let physical_block = bt_data[b * max_num_blocks + logical_block] as usize;
            for kv_h in 0..num_kv_heads {
                let src = cache_offset(
                    physical_block,
                    block_offset,
                    kv_h,
                    num_kv_heads,
                    head_dim,
                    block_size,
                );
                let dst = ((b * num_kv_heads + kv_h) * seq_len_k + t) * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&kv_data[src..src + head_dim]);
            }
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        &[batch_size, num_kv_heads, seq_len_k, head_dim],
        kv_blocks.device(),
    )?)
}

/// Scatter contiguous gradients `[B, num_kv_heads, seq_len_k, head_dim]` back to
/// the paged block layout.
///
/// Accumulates rather than overwrites: two sequences may share a physical
/// block, and a gradient reaching the same page twice must sum.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_to_paged(
    grad_cont: &Tensor<CpuRuntime>,
    kv_blocks_ref: &Tensor<CpuRuntime>,
    block_table: &Tensor<CpuRuntime>,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor<CpuRuntime>> {
    let grad_data = grad_cont.to_vec::<f32>();
    let bt_data = block_table.to_vec::<i32>();
    let max_num_blocks = block_table.shape()[1];
    let block_shape = kv_blocks_ref.shape();

    let total_blocks = block_shape[0];
    let mut out = vec![0.0f32; total_blocks * block_size * num_kv_heads * head_dim];

    for b in 0..batch_size {
        for t in 0..seq_len_k {
            let logical_block = t / block_size;
            let block_offset = t % block_size;
            let physical_block = bt_data[b * max_num_blocks + logical_block] as usize;
            for kv_h in 0..num_kv_heads {
                let dst = cache_offset(
                    physical_block,
                    block_offset,
                    kv_h,
                    num_kv_heads,
                    head_dim,
                    block_size,
                );
                let src = ((b * num_kv_heads + kv_h) * seq_len_k + t) * head_dim;
                for d in 0..head_dim {
                    out[dst + d] += grad_data[src + d];
                }
            }
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        block_shape,
        kv_blocks_ref.device(),
    )?)
}

/// Expand a gathered `[B, num_kv_heads, S, D]` tensor to `[B, num_heads, S, D]`,
/// repeating each KV head for the query heads that share it.
///
/// Query head `h` reads KV head `h / (num_heads / num_kv_heads)`, so the repeat
/// must be interleaved rather than tiled.
pub(super) fn expand_kv_heads(
    client: &CpuClient,
    kv: &Tensor<CpuRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<Tensor<CpuRuntime>> {
    use crate::error::Error;
    use numr::ops::ShapeOps;

    if num_kv_heads == num_heads {
        return Ok(kv.clone());
    }
    client
        .repeat_interleave(kv, num_heads / num_kv_heads, Some(1))
        .map_err(Error::Numr)
}

/// Reduce a `[B, num_heads, S, D]` gradient back to `[B, num_kv_heads, S, D]`,
/// summing the query heads that shared each KV head.
///
/// The inverse of [`expand_kv_heads`]: the shared heads are adjacent, so the
/// group axis splits out by reshape and sums away.
pub(super) fn reduce_kv_heads(
    client: &CpuClient,
    grad: &Tensor<CpuRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_k: usize,
    head_dim: usize,
) -> Result<Tensor<CpuRuntime>> {
    use crate::error::Error;
    use numr::ops::ReduceOps;

    if num_kv_heads == num_heads {
        return Ok(grad.clone());
    }
    let grouped = grad
        .reshape(&[
            batch_size,
            num_kv_heads,
            num_heads / num_kv_heads,
            seq_len_k,
            head_dim,
        ])
        .map_err(Error::Numr)?;
    client.sum(&grouped, &[2], false).map_err(Error::Numr)
}
