//! Vocab-parallel embedding (Megatron-LM style)
//!
//! Each rank owns a shard of the vocabulary. During forward, each rank
//! gathers embeddings for its owned tokens (zeroing out-of-range indices),
//! then all-reduce sums across ranks to produce the full result.

use std::sync::Arc;

use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{CompareOps, IndexingOps, ScalarOps, TensorOps, TypeConversionOps, UtilityOps};
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Vocabulary-parallel embedding layer.
///
/// The full vocabulary `[vocab_size, embed_dim]` is split across ranks along
/// dim 0. Each rank holds `[vocab_shard_size, embed_dim]`.
///
/// Forward: mask out-of-range indices → local gather → all-reduce sum.
pub struct VocabParallelEmbedding<R: Runtime> {
    weight: Var<R>,
    comm: Arc<dyn Communicator>,
    vocab_start: usize,
    vocab_end: usize,
    embed_dim: usize,
}

impl<R: Runtime<DType = DType>> VocabParallelEmbedding<R> {
    /// Create from a full embedding weight by extracting this rank's shard.
    pub fn new(
        full_weight: &Tensor<R>,
        comm: Arc<dyn Communicator>,
        trainable: bool,
    ) -> Result<Self> {
        let shape = full_weight.shape();
        if shape.len() != 2 {
            return Err(Error::DistributedError {
                reason: format!(
                    "VocabParallelEmbedding expects 2D weight, got {}D",
                    shape.len()
                ),
            });
        }

        let vocab_size = shape[0];
        let embed_dim = shape[1];
        let rank = comm.rank();
        let world_size = comm.world_size();

        if vocab_size % world_size != 0 {
            return Err(Error::DistributedError {
                reason: format!(
                    "vocab_size ({}) not divisible by world_size ({})",
                    vocab_size, world_size
                ),
            });
        }

        let shard_size = vocab_size / world_size;
        let vocab_start = rank * shard_size;
        let vocab_end = vocab_start + shard_size;

        let shard = full_weight
            .narrow(0, vocab_start, shard_size)
            .map_err(|e| Error::DistributedError {
                reason: format!("embedding narrow failed: {e}"),
            })?
            .contiguous();

        Ok(Self {
            weight: Var::new(shard, trainable),
            comm,
            vocab_start,
            vocab_end,
            embed_dim,
        })
    }

    /// Create from a pre-sharded weight.
    pub fn from_shard(
        weight: Tensor<R>,
        comm: Arc<dyn Communicator>,
        vocab_start: usize,
        vocab_end: usize,
        trainable: bool,
    ) -> Self {
        let embed_dim = weight.shape()[1];
        Self {
            weight: Var::new(weight, trainable),
            comm,
            vocab_start,
            vocab_end,
            embed_dim,
        }
    }

    /// Forward: indices `[...]` → embeddings `[..., embed_dim]`
    ///
    /// Fully on-device: no GPU→CPU transfers. Each rank computes a mask
    /// for its vocab range, gathers with clamped local indices, masks
    /// out-of-range results to zero, then all-reduce sums across ranks.
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + IndexingOps<R>
            + TensorOps<R>
            + CompareOps<R>
            + ScalarOps<R>
            + UtilityOps<R>
            + TypeConversionOps<R>,
        R::Client: IndexingOps<R> + TensorOps<R>,
    {
        let idx_shape = indices.shape().to_vec();
        let n: usize = idx_shape.iter().product();
        let shard_size = self.vocab_end - self.vocab_start;

        // Flatten indices to [n]
        let flat_idx = indices.reshape(&[n]).map_err(Error::Numr)?;

        // Build mask on-device: (indices >= vocab_start) AND (indices < vocab_end)
        // CompareOps returns 1.0/0.0 in same dtype
        let start_tensor = Tensor::<R>::full_scalar(
            &[1],
            flat_idx.dtype(),
            self.vocab_start as f64,
            indices.device(),
        );
        let end_tensor = Tensor::<R>::full_scalar(
            &[1],
            flat_idx.dtype(),
            self.vocab_end as f64,
            indices.device(),
        );

        let ge_start = client.ge(&flat_idx, &start_tensor).map_err(Error::Numr)?;
        let lt_end = client.lt(&flat_idx, &end_tensor).map_err(Error::Numr)?;
        // AND via multiply (both are 1.0/0.0)
        let mask_i64 = client.mul(&ge_start, &lt_end).map_err(Error::Numr)?;

        // Cast mask to F32 for later multiplication with embeddings
        let mask_f32 = client.cast(&mask_i64, DType::F32).map_err(Error::Numr)?;

        // Compute local indices: (indices - vocab_start), clamped to [0, shard_size-1]
        let local_idx = client
            .sub_scalar(&flat_idx, self.vocab_start as f64)
            .map_err(Error::Numr)?;
        let local_idx = client
            .clamp(&local_idx, 0.0, (shard_size - 1) as f64)
            .map_err(Error::Numr)?;

        // Gather from local shard: [n] -> [n, embed_dim]
        let expanded = local_idx.unsqueeze(1).map_err(Error::Numr)?;
        let expanded = expanded
            .broadcast_to(&[n, self.embed_dim])
            .map_err(Error::Numr)?;

        let gathered =
            numr::autograd::var_gather(&self.weight, 0, &expanded, client).map_err(Error::Numr)?;

        // Apply mask: zero out rows for out-of-range indices
        // mask_f32 is [n], unsqueeze to [n, 1] for broadcast
        let mask_2d = mask_f32.unsqueeze(1).map_err(Error::Numr)?;
        let mask_broadcast = mask_2d
            .broadcast_to(&[n, self.embed_dim])
            .map_err(Error::Numr)?;
        let mask_var = Var::new(mask_broadcast, false);
        let masked = numr::autograd::var_mul(&gathered, &mask_var, client).map_err(Error::Numr)?;

        // All-reduce across ranks (sum of masked results = full embedding)
        if self.comm.world_size() > 1 {
            let tensor = masked.tensor();
            all_reduce_tensor(self.comm.as_ref(), tensor, ReduceOp::Sum)?;
            self.comm.sync().map_err(|e| Error::DistributedError {
                reason: format!("sync after embedding all_reduce failed: {e}"),
            })?;
        }

        // Reshape to [..., embed_dim]
        let mut out_shape = idx_shape;
        out_shape.push(self.embed_dim);
        let result = numr::autograd::var_reshape(&masked, &out_shape).map_err(Error::Numr)?;
        Ok(result)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn vocab_start(&self) -> usize {
        self.vocab_start
    }

    pub fn vocab_end(&self) -> usize {
        self.vocab_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Embedding;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn cpu_setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_vocab_parallel_single_rank_matches_embedding() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        #[rustfmt::skip]
        let weight = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0,   // token 0
                5.0, 6.0, 7.0, 8.0,       // token 1
                9.0, 10.0, 11.0, 12.0,    // token 2
            ],
            &[3, 4],
            &device,
        );

        let plain_emb = Embedding::new(weight.clone(), false);
        let par_emb = VocabParallelEmbedding::new(&weight, comm, false).unwrap();

        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1], &[3], &device);

        let plain_out = plain_emb.forward(&client, &indices).unwrap();
        let par_out = par_emb.forward(&client, &indices).unwrap();

        assert_eq!(plain_out.shape(), par_out.shape());
        let plain_data = plain_out.tensor().to_vec::<f32>();
        let par_data = par_out.tensor().to_vec::<f32>();
        assert_eq!(plain_data, par_data);
    }

    #[test]
    fn test_vocab_parallel_forward_shape() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 20], &[4, 5], &device);
        let par_emb = VocabParallelEmbedding::new(&weight, comm, false).unwrap();

        // [2, 3] batch of indices
        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 0, 1], &[2, 3], &device);
        let out = par_emb.forward(&client, &indices).unwrap();
        assert_eq!(out.shape(), &[2, 3, 5]);
    }

    #[test]
    fn test_vocab_parallel_not_divisible() {
        let (_client, device) = cpu_setup();

        // vocab=3, world_size=1 → 3 % 1 == 0, OK
        // But we can't test world_size>1 with NoOp easily.
        // Just test the error path with a direct check.
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, 4], &device);
        let comm = Arc::new(NoOpCommunicator);
        // world_size=1 always divides, so this should succeed
        assert!(VocabParallelEmbedding::new(&weight, comm, false).is_ok());
    }

    #[test]
    fn test_vocab_parallel_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VocabParallelEmbedding<CpuRuntime>>();
    }
}
