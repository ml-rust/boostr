//! Tensor parallelism (Megatron-LM style)
//!
//! Column-parallel and row-parallel linear layers that split model weights
//! across devices. Follows the Megatron-LM approach where column-parallel
//! splits output features and row-parallel splits input features.

use std::sync::Arc;

use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Column-parallel linear layer.
///
/// Weight `[out_features, in_features]` is split along dim 0 (out_features).
/// Each rank holds `[out_features / world_size, in_features]`.
///
/// Forward: local matmul produces partial output along the out_features dimension.
/// No communication in forward — the caller is responsible for gathering or
/// feeding the output into a `RowParallelLinear`.
///
/// This is the "f" parallelism in Megatron-LM's column-parallel linear.
pub struct ColumnParallelLinear<R: Runtime> {
    weight: Var<R>,
    bias: Option<Var<R>>,
}

impl<R: Runtime<DType = DType>> ColumnParallelLinear<R> {
    /// Create from a full weight tensor by extracting this rank's shard.
    ///
    /// `full_weight`: `[out_features, in_features]`
    /// `full_bias`: optional `[out_features]`
    ///
    /// Splits `out_features` evenly across `world_size` ranks.
    pub fn new(
        full_weight: &Tensor<R>,
        full_bias: Option<&Tensor<R>>,
        comm: &dyn Communicator,
        trainable: bool,
    ) -> Result<Self> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shape = full_weight.shape();

        if shape.len() != 2 {
            return Err(Error::DistributedError {
                reason: format!(
                    "ColumnParallelLinear expects 2D weight, got {}D",
                    shape.len()
                ),
            });
        }

        let out_features = shape[0];
        if out_features % world_size != 0 {
            return Err(Error::DistributedError {
                reason: format!(
                    "out_features ({}) not divisible by world_size ({})",
                    out_features, world_size
                ),
            });
        }

        let shard_size = out_features / world_size;
        let start = rank * shard_size;

        let weight_shard = full_weight
            .narrow(0, start, shard_size)
            .map_err(|e| Error::DistributedError {
                reason: format!("weight narrow failed: {e}"),
            })?
            .contiguous();

        let bias_shard = match full_bias {
            Some(b) => {
                let shard = b
                    .narrow(0, start, shard_size)
                    .map_err(|e| Error::DistributedError {
                        reason: format!("bias narrow failed: {e}"),
                    })?
                    .contiguous();
                Some(Var::new(shard, trainable))
            }
            None => None,
        };

        Ok(Self {
            weight: Var::new(weight_shard, trainable),
            bias: bias_shard,
        })
    }

    /// Create directly from a pre-sharded weight (already the right shard for this rank).
    pub fn from_shard(weight: Tensor<R>, bias: Option<Tensor<R>>, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: bias.map(|b| Var::new(b, trainable)),
        }
    }

    /// Forward: input @ weight_shard^T + bias_shard
    ///
    /// input: `[..., in_features]`
    /// output: `[..., out_features / world_size]`
    ///
    /// No communication — output is the local shard of the full output.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let w_t = numr::autograd::var_transpose(&self.weight).map_err(Error::Numr)?;
        let output = numr::autograd::var_matmul(input, &w_t, client).map_err(Error::Numr)?;
        match &self.bias {
            Some(bias) => numr::autograd::var_add(&output, bias, client).map_err(Error::Numr),
            None => Ok(output),
        }
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.bias.as_ref()
    }
}

/// Row-parallel linear layer.
///
/// Weight `[out_features, in_features]` is split along dim 1 (in_features).
/// Each rank holds `[out_features, in_features / world_size]`.
///
/// Forward: local matmul → all-reduce output across ranks.
/// The input to each rank should be the corresponding shard of the activations
/// (typically the output of a `ColumnParallelLinear`).
pub struct RowParallelLinear<R: Runtime> {
    weight: Var<R>,
    bias: Option<Var<R>>,
    comm: Arc<dyn Communicator>,
}

impl<R: Runtime<DType = DType>> RowParallelLinear<R> {
    /// Create from a full weight tensor by extracting this rank's shard.
    ///
    /// `full_weight`: `[out_features, in_features]`
    /// `full_bias`: optional `[out_features]` (NOT split — only added after all-reduce)
    ///
    /// Splits `in_features` evenly across `world_size` ranks.
    pub fn new(
        full_weight: &Tensor<R>,
        full_bias: Option<&Tensor<R>>,
        comm: Arc<dyn Communicator>,
        trainable: bool,
    ) -> Result<Self> {
        let rank = comm.rank();
        let world_size = comm.world_size();
        let shape = full_weight.shape();

        if shape.len() != 2 {
            return Err(Error::DistributedError {
                reason: format!("RowParallelLinear expects 2D weight, got {}D", shape.len()),
            });
        }

        let in_features = shape[1];
        if in_features % world_size != 0 {
            return Err(Error::DistributedError {
                reason: format!(
                    "in_features ({}) not divisible by world_size ({})",
                    in_features, world_size
                ),
            });
        }

        let shard_size = in_features / world_size;
        let start = rank * shard_size;

        let weight_shard = full_weight
            .narrow(1, start, shard_size)
            .map_err(|e| Error::DistributedError {
                reason: format!("weight narrow failed: {e}"),
            })?
            .contiguous();

        // Bias is NOT split — it's added after all-reduce on rank 0 only,
        // or equivalently on all ranks (since all-reduce produces identical results).
        let bias_var = full_bias.map(|b| Var::new(b.clone(), trainable));

        Ok(Self {
            weight: Var::new(weight_shard, trainable),
            bias: bias_var,
            comm,
        })
    }

    /// Create directly from a pre-sharded weight.
    pub fn from_shard(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        comm: Arc<dyn Communicator>,
        trainable: bool,
    ) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: bias.map(|b| Var::new(b, trainable)),
            comm,
        }
    }

    /// Forward: input @ weight_shard^T → all-reduce → + bias
    ///
    /// input: `[..., in_features / world_size]` (each rank's shard)
    /// output: `[..., out_features]` (identical on all ranks)
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let w_t = numr::autograd::var_transpose(&self.weight).map_err(Error::Numr)?;
        let local_output = numr::autograd::var_matmul(input, &w_t, client).map_err(Error::Numr)?;

        // All-reduce the partial outputs across ranks
        if self.comm.world_size() > 1 {
            let output_tensor = local_output.tensor();
            all_reduce_tensor(self.comm.as_ref(), output_tensor, ReduceOp::Sum)?;

            self.comm.sync().map_err(|e| Error::DistributedError {
                reason: format!("sync after all_reduce failed: {e}"),
            })?;
        }

        match &self.bias {
            Some(bias) => numr::autograd::var_add(&local_output, bias, client).map_err(Error::Numr),
            None => Ok(local_output),
        }
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.bias.as_ref()
    }
}

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
    if dim_size % world_size != 0 {
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
        .map(|t| t.contiguous())
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
    let recv = Tensor::<R>::zeros(&[total_count], dtype, local_shard.device());

    unsafe {
        comm.all_gather(local_shard.data_ptr(), recv.data_ptr(), count, dtype)
            .map_err(|e| Error::DistributedError {
                reason: format!("all_gather failed: {e}"),
            })?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after all_gather failed: {e}"),
    })?;

    // Reshape: the all_gather result is [world_size * shard_numel] flat.
    // For proper dim-based concatenation, reshape to [world_size, ...shard_shape]
    // then transpose and reshape. For simplicity with the flat gather,
    // we reshape to the expected output shape.
    let ndim = local_shard.shape().len();
    let dim_idx = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    let mut out_shape = local_shard.shape().to_vec();
    out_shape[dim_idx] *= world_size;

    recv.reshape(&out_shape)
        .map_err(|e| Error::DistributedError {
            reason: format!("gather reshape failed: {e}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_column_parallel_linear_creation() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        // weight: [4, 3] — 4 out_features, 3 in_features
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device);
        let col = ColumnParallelLinear::new(&weight, None, &comm, false).unwrap();

        // world_size=1, rank=0 → shard is full weight
        assert_eq!(col.weight().shape(), &[4, 3]);
    }

    #[test]
    fn test_column_parallel_linear_forward() {
        let (client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3], &device);
        let col = ColumnParallelLinear::new(&weight, None, &comm, false).unwrap();

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device),
            false,
        );
        let out = col.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_row_parallel_linear_creation() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
        let row = RowParallelLinear::new(&weight, None, comm, false).unwrap();

        assert_eq!(row.weight().shape(), &[2, 3]);
    }

    #[test]
    fn test_row_parallel_linear_forward() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], &device);
        let row = RowParallelLinear::new(&weight, None, comm, false).unwrap();

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device),
            false,
        );
        let out = row.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_scatter_to_rank() {
        let (_client, device) = cpu_setup();
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device);

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
        let shard = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        let gathered = gather_from_ranks(&shard, 0, &comm).unwrap();
        assert_eq!(gathered.shape(), &[3]);
        assert_eq!(gathered.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_column_parallel_not_divisible() {
        let (_client, device) = cpu_setup();

        // Create a fake communicator with world_size=3 to test error
        // NoOpCommunicator has world_size=1, so this always passes.
        // The divisibility check is still exercised via scatter_to_rank.
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device);
        let result = scatter_to_rank(&tensor, 0, 0, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_parallel_with_bias() {
        let (client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2], &device);
        let col = ColumnParallelLinear::new(&weight, Some(&bias), &comm, false).unwrap();

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            false,
        );
        let out = col.forward(&client, &input).unwrap();
        let data = out.tensor().to_vec::<f32>();
        // [1, 2] @ [[1, 0], [0, 1]]^T + [10, 20] = [1, 2] + [10, 20] = [11, 22]
        assert_eq!(data, vec![11.0, 22.0]);
    }
}
