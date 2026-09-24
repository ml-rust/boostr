//! Paged KV cache layout mapping for the CPU backend.
//!
//! The cache is `[num_blocks, block_size, num_kv_heads, head_dim]` and the
//! block table maps a sequence's logical block index to a physical one, per the
//! layout contract on `PagedAttentionOps`. These two functions are the only
//! place that mapping is written down on CPU; the attention itself runs on the
//! gathered contiguous tensors.

use crate::error::{Error, Result};
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
    let bt_data = block_table.to_vec::<i32>();
    let max_num_blocks = block_table.shape()[1];
    let dtype = kv_blocks.dtype();
    let device = kv_blocks.device();
    let out_shape = [batch_size, num_kv_heads, seq_len_k, head_dim];

    // The dtype-erroring arms of `dispatch_dtype!` (unsupported/feature-gated
    // dtype) return numr's own error type, so they run inside a closure and
    // get mapped to boostr's `Error` at the boundary rather than trying to
    // `return` numr's error type out of this function directly.
    let gathered = (|| -> numr::error::Result<Tensor<CpuRuntime>> {
        numr::dispatch_dtype!(dtype, T => {
            let kv_data = kv_blocks.to_vec::<T>();
            let mut out = vec![T::default(); batch_size * num_kv_heads * seq_len_k * head_dim];

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

            Tensor::<CpuRuntime>::from_slice(&out, &out_shape, device)
        }, "gather_paged_kv")
    })();

    gathered.map_err(Error::Numr)
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
    let bt_data = block_table.to_vec::<i32>();
    let max_num_blocks = block_table.shape()[1];
    let block_shape = kv_blocks_ref.shape();
    let dtype = kv_blocks_ref.dtype();
    let device = kv_blocks_ref.device();

    if grad_cont.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg: "grad_cont",
            reason: format!(
                "grad_cont has dtype {:?} but kv_blocks_ref has dtype {:?}; scatter_to_paged requires both to share one dtype",
                grad_cont.dtype(),
                dtype
            ),
        });
    }

    let total_blocks = block_shape[0];

    // See `gather_paged_kv` for why the macro runs inside a closure: its
    // unsupported/feature-gated dtype arms return numr's own error type.
    let scattered = (|| -> numr::error::Result<Tensor<CpuRuntime>> {
        numr::dispatch_dtype!(dtype, T => {
            let grad_data = grad_cont.to_vec::<T>();
            let mut out = vec![T::default(); total_blocks * block_size * num_kv_heads * head_dim];

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

            Tensor::<CpuRuntime>::from_slice(&out, block_shape, device)
        }, "scatter_to_paged")
    })();

    scattered.map_err(Error::Numr)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuDevice;

    fn block_table(device: &CpuDevice, data: &[i32], shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
    }

    /// `[num_blocks=2, block_size=2, num_kv_heads=1, head_dim=2]` cache with
    /// values `1..=8`, and a block table that reads blocks `[0, 1]` in order,
    /// so the gathered `[1, 1, 4, 2]` output is `1..=8` unpermuted.
    #[test]
    fn test_gather_paged_kv_f32_round_trip() {
        let (_client, device) = cpu_setup();
        let cache = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 1, 2],
            &device,
        )
        .unwrap();
        let bt = block_table(&device, &[0, 1], &[1, 2]);

        let out = gather_paged_kv(&cache, &bt, 1, 1, 4, 2, 2).unwrap();

        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(
            out.to_vec::<f32>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_gather_paged_kv_bf16_round_trip() {
        use half::bf16;

        let (_client, device) = cpu_setup();
        let vals: Vec<bf16> = (1..=8).map(|v| bf16::from_f32(v as f32)).collect();
        let cache = Tensor::<CpuRuntime>::from_slice(&vals, &[2, 2, 1, 2], &device).unwrap();
        let bt = block_table(&device, &[0, 1], &[1, 2]);

        let out = gather_paged_kv(&cache, &bt, 1, 1, 4, 2, 2).unwrap();

        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.to_vec::<bf16>(), vals);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_gather_paged_kv_f16_round_trip() {
        use half::f16;

        let (_client, device) = cpu_setup();
        let vals: Vec<f16> = (1..=8).map(|v| f16::from_f32(v as f32)).collect();
        let cache = Tensor::<CpuRuntime>::from_slice(&vals, &[2, 2, 1, 2], &device).unwrap();
        let bt = block_table(&device, &[0, 1], &[1, 2]);

        let out = gather_paged_kv(&cache, &bt, 1, 1, 4, 2, 2).unwrap();

        assert_eq!(out.dtype(), DType::F16);
        assert_eq!(out.to_vec::<f16>(), vals);
    }

    #[test]
    fn test_scatter_to_paged_f32_preserves_dtype_and_sums() {
        let (_client, device) = cpu_setup();
        let kv_blocks_ref =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 2, 1, 2], &device).unwrap();
        // Both batch entries hit token 0 of the sequence (physical block 0,
        // offset 0), so their gradients must accumulate rather than overwrite.
        let grad = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
            &[2, 1, 2, 2],
            &device,
        )
        .unwrap();
        let bt = block_table(&device, &[0, 1, 0, 1], &[2, 2]);

        let out = scatter_to_paged(&grad, &kv_blocks_ref, &bt, 2, 1, 2, 2, 2).unwrap();

        assert_eq!(out.dtype(), DType::F32);
        let data = out.to_vec::<f32>();
        assert_eq!(&data[0..2], &[3.0, 3.0]); // block 0, offset 0: 1+2
    }

    #[test]
    fn test_scatter_to_paged_dtype_mismatch_is_error() {
        let (_client, device) = cpu_setup();
        let kv_blocks_ref =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 2, 1, 2], &device).unwrap();
        let grad = Tensor::<CpuRuntime>::from_slice(&[1i32; 8], &[2, 1, 2, 2], &device).unwrap();
        let bt = block_table(&device, &[0, 1, 0, 1], &[2, 2]);

        let err = scatter_to_paged(&grad, &kv_blocks_ref, &bt, 2, 1, 2, 2, 2).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("F32"), "expected F32 named in error: {msg}");
        assert!(msg.contains("I32"), "expected I32 named in error: {msg}");
    }
}
