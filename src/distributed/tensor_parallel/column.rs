//! Column-parallel linear layer: the weight is split along `out_features`.

use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
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
        if !out_features.is_multiple_of(world_size) {
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
            .contiguous()
            .map_err(Error::Numr)?;

        let bias_shard = match full_bias {
            Some(b) => {
                let shard = b
                    .narrow(0, start, shard_size)
                    .map_err(|e| Error::DistributedError {
                        reason: format!("bias narrow failed: {e}"),
                    })?
                    .contiguous()
                    .map_err(Error::Numr)?;
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
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let col = ColumnParallelLinear::new(&weight, None, &comm, false).unwrap();

        // world_size=1, rank=0 → shard is full weight
        assert_eq!(col.weight().shape(), &[4, 3]);
    }

    #[test]
    fn test_column_parallel_linear_forward() {
        let (client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3], &device)
                .unwrap();
        let col = ColumnParallelLinear::new(&weight, None, &comm, false).unwrap();

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap(),
            false,
        );
        let out = col.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_column_parallel_with_bias() {
        let (client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2], &device).unwrap();
        let col = ColumnParallelLinear::new(&weight, Some(&bias), &comm, false).unwrap();

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
            false,
        );
        let out = col.forward(&client, &input).unwrap();
        let data = out.tensor().to_vec::<f32>();
        // [1, 2] @ [[1, 0], [0, 1]]^T + [10, 20] = [1, 2] + [10, 20] = [11, 22]
        assert_eq!(data, vec![11.0, 22.0]);
    }
}
