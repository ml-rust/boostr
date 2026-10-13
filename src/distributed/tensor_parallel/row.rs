//! Row-parallel linear layer: the weight is split along `in_features`, and
//! the partial outputs are all-reduced.

use std::sync::Arc;

use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
        if !in_features.is_multiple_of(world_size) {
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
            .contiguous()
            .map_err(Error::Numr)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_row_parallel_linear_creation() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
        let row = RowParallelLinear::new(&weight, None, comm, false).unwrap();

        assert_eq!(row.weight().shape(), &[2, 3]);
    }

    #[test]
    fn test_row_parallel_linear_forward() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], &device)
                .unwrap();
        let row = RowParallelLinear::new(&weight, None, comm, false).unwrap();

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap(),
            false,
        );
        let out = row.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
    }
}
