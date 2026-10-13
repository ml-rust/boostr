//! Shard helpers: split a tensor across ranks along one dimension, and
//! all-gather the shards back into the full tensor.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::{Communicator, Runtime};
use numr::tensor::Tensor;

/// Split a tensor evenly across ranks along a given dimension.
///
/// Returns the shard for the given rank.
pub fn scatter_to_rank<R: Runtime>(
    tensor: &Tensor<R>,
    dim: isize,
    rank: usize,
    world_size: usize,
) -> Result<Tensor<R>> {
    let ndim = tensor.shape().len();
    let dim_idx = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if dim_idx >= ndim {
        return Err(Error::DistributedError {
            reason: format!("dim {} out of range for {}D tensor", dim, ndim),
        });
    }

    let dim_size = tensor.shape()[dim_idx];
    if !dim_size.is_multiple_of(world_size) {
        return Err(Error::DistributedError {
            reason: format!(
                "dim {} size ({}) not divisible by world_size ({})",
                dim, dim_size, world_size
            ),
        });
    }

    let shard_size = dim_size / world_size;
    let start = rank * shard_size;

    tensor
        .narrow(dim, start, shard_size)
        .and_then(|t| t.contiguous())
        .map_err(|e| Error::DistributedError {
            reason: format!("scatter narrow failed: {e}"),
        })
}

/// Gather shards from all ranks along a given dimension using all_gather.
///
/// Each rank provides its local shard. After this call, all ranks have the
/// full concatenated tensor.
pub fn gather_from_ranks<R: Runtime<DType = DType>>(
    local_shard: &Tensor<R>,
    dim: isize,
    comm: &dyn Communicator,
) -> Result<Tensor<R>> {
    let world_size = comm.world_size();

    if world_size <= 1 {
        return Ok(local_shard.clone());
    }

    if !local_shard.is_contiguous() {
        return Err(Error::DistributedError {
            reason: "gather_from_ranks requires contiguous tensors".to_string(),
        });
    }

    let count = local_shard.numel();
    let dtype = local_shard.dtype();
    let total_count = count * world_size;

    // Allocate receive buffer
    let recv = Tensor::<R>::zeros(&[total_count], dtype, local_shard.device())?;

    unsafe {
        comm.all_gather(local_shard.ptr(), recv.ptr(), count, dtype)
            .map_err(|e| Error::DistributedError {
                reason: format!("all_gather failed: {e}"),
            })?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after all_gather failed: {e}"),
    })?;

    concat_gathered(recv, local_shard.shape(), dim, world_size)
}

/// Turn a flat all-gather buffer into shards concatenated along `dim`.
///
/// The buffer holds `world_size` shards back to back, rank 0 first. It views
/// as `[world_size, ...shard_shape]`. Rank axis 0 moves next to `dim`, then
/// the two axes merge.
fn concat_gathered<R: Runtime<DType = DType>>(
    flat: Tensor<R>,
    shard_shape: &[usize],
    dim: isize,
    world_size: usize,
) -> Result<Tensor<R>> {
    let ndim = shard_shape.len();
    let dim_idx = if dim < 0 { ndim as isize + dim } else { dim };
    if dim_idx < 0 || dim_idx as usize >= ndim {
        return Err(Error::DistributedError {
            reason: format!("gather dim {dim} is out of range for a {ndim}-d shard"),
        });
    }
    let dim_idx = dim_idx as usize;
    let gather_err = |e: numr::error::Error| Error::DistributedError {
        reason: format!("gather reshape failed: {e}"),
    };

    let mut stacked_shape = Vec::with_capacity(ndim + 1);
    stacked_shape.push(world_size);
    stacked_shape.extend_from_slice(shard_shape);
    let stacked = flat.reshape(&stacked_shape).map_err(gather_err)?;

    // [W, d0.., dk, ..] -> [d0.., W, dk, ..]: the rank axis lands before `dim`.
    let mut order: Vec<usize> = (1..=dim_idx).collect();
    order.push(0);
    order.extend(dim_idx + 1..=ndim);
    let moved = stacked
        .permute(&order)
        .and_then(|t| t.contiguous())
        .map_err(gather_err)?;

    let mut out_shape = shard_shape.to_vec();
    out_shape[dim_idx] *= world_size;
    moved.reshape(&out_shape).map_err(gather_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_scatter_to_rank() {
        let (_client, device) = cpu_setup();
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();

        // world_size=1
        let shard = scatter_to_rank(&tensor, 0, 0, 1).unwrap();
        assert_eq!(shard.shape(), &[4, 3]);

        // world_size=2
        let shard0 = scatter_to_rank(&tensor, 0, 0, 2).unwrap();
        let shard1 = scatter_to_rank(&tensor, 0, 1, 2).unwrap();
        assert_eq!(shard0.shape(), &[2, 3]);
        assert_eq!(shard1.shape(), &[2, 3]);
    }

    #[test]
    fn test_gather_from_ranks_noop() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;
        let shard = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        let gathered = gather_from_ranks(&shard, 0, &comm).unwrap();
        assert_eq!(gathered.shape(), &[3]);
        assert_eq!(gathered.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_concat_gathered_last_dim() {
        let (_client, device) = cpu_setup();
        // Two ranks, each shard [2, 2]: rank 0 = [[0,1],[2,3]], rank 1 = [[4,5],[6,7]].
        let flat = Tensor::<CpuRuntime>::from_slice(
            &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[8],
            &device,
        )
        .unwrap();
        let out = concat_gathered(flat.clone(), &[2, 2], -1, 2).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(
            out.to_vec::<f32>(),
            vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]
        );

        let out = concat_gathered(flat, &[2, 2], 0, 2).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(
            out.to_vec::<f32>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn test_concat_gathered_rejects_bad_dim() {
        let (_client, device) = cpu_setup();
        let flat = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        assert!(concat_gathered(flat, &[2], 1, 2).is_err());
    }

    #[test]
    fn test_column_parallel_not_divisible() {
        let (_client, device) = cpu_setup();

        // Create a fake communicator with world_size=3 to test error
        // NoOpCommunicator has world_size=1, so this always passes.
        // The divisibility check is still exercised via scatter_to_rank.
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let result = scatter_to_rank(&tensor, 0, 0, 3);
        assert!(result.is_err());
    }
}
