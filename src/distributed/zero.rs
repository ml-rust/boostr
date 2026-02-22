//! ZeRO Stage 1 optimizer — shards optimizer state across data-parallel ranks
//!
//! Each rank only stores AdamW `m` and `v` tensors for ~1/N of parameters.
//! After the optimizer step, each rank broadcasts its updated parameters so
//! all ranks have the full model. This reduces optimizer memory by N.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::distributed::comm_utils::broadcast_tensor;
use crate::error::{Error, Result};
use crate::optimizer::{AdamW, AdamWConfig};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// ZeRO Stage 1: optimizer state partitioning across data-parallel ranks.
///
/// Wraps `AdamW` and restricts each rank to only update (and store m/v for)
/// its owned subset of parameters. After the optimizer step, updated params
/// are broadcast from each owner to all other ranks.
///
/// With `world_size=1`, this degenerates to plain `AdamW` with no communication.
pub struct ZeroStage1<R: Runtime> {
    optimizer: AdamW<R>,
    comm: Arc<dyn Communicator>,
    world_size: usize,
    /// Parameter IDs this rank owns (and stores optimizer state for).
    /// Fast lookup set; `param_owners` maintains the ordered list for broadcasts.
    owned_params: HashSet<TensorId>,
    /// All param IDs in deterministic order, with their owning rank
    param_owners: Vec<(TensorId, usize)>,
}

impl<R: Runtime<DType = DType>> ZeroStage1<R> {
    /// Create a new ZeRO Stage 1 optimizer.
    ///
    /// Parameters are assigned round-robin by sorted TensorId to ranks.
    /// Each rank will only store optimizer state for its owned params.
    ///
    /// # Arguments
    /// * `config` - AdamW configuration
    /// * `comm` - Communicator for broadcast operations
    /// * `param_ids` - All parameter IDs (must be the same on all ranks)
    pub fn new(config: AdamWConfig, comm: Arc<dyn Communicator>, param_ids: &[TensorId]) -> Self {
        let rank = comm.rank();
        let world_size = comm.world_size();

        // Sort for deterministic assignment across ranks
        let mut sorted_ids: Vec<TensorId> = param_ids.to_vec();
        sorted_ids.sort_by_key(|id| id.raw());

        let mut owned_params = HashSet::new();
        let mut param_owners = Vec::with_capacity(sorted_ids.len());

        for (i, &id) in sorted_ids.iter().enumerate() {
            let owner = i % world_size;
            param_owners.push((id, owner));
            if owner == rank {
                owned_params.insert(id);
            }
        }

        Self {
            optimizer: AdamW::new(config),
            comm,
            world_size,
            owned_params,
            param_owners,
        }
    }

    /// Perform one ZeRO Stage 1 optimizer step.
    ///
    /// 1. Each rank runs AdamW only on its owned params
    /// 2. Each rank broadcasts its updated params to all other ranks
    /// 3. All ranks end up with identical, fully-updated parameters
    pub fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R>,
    {
        if self.world_size <= 1 {
            // Single rank: plain AdamW, no communication
            return self.optimizer.step(client, params, grads);
        }

        // Step 1: Move owned params into a filtered map (no cloning)
        let mut owned_param_map: HashMap<TensorId, Tensor<R>> = HashMap::new();
        for &id in &self.owned_params {
            if let Some(t) = params.remove(&id) {
                owned_param_map.insert(id, t);
            }
        }

        // Run AdamW only on owned params
        self.optimizer.step(client, &mut owned_param_map, grads)?;

        // Move updated params back into the full params map
        for (id, tensor) in owned_param_map {
            params.insert(id, tensor);
        }

        // Step 2: Broadcast updated params from each owner to all ranks
        for &(id, owner) in &self.param_owners {
            if let Some(tensor) = params.get(&id) {
                broadcast_tensor(self.comm.as_ref(), tensor, owner)?;
            }
        }

        self.comm.sync().map_err(|e| Error::DistributedError {
            reason: format!("sync after ZeRO broadcast failed: {e}"),
        })?;

        Ok(())
    }

    /// The set of parameter IDs this rank owns.
    pub fn owned_param_ids(&self) -> &HashSet<TensorId> {
        &self.owned_params
    }

    /// Current optimizer timestep.
    pub fn timestep(&self) -> u64 {
        self.optimizer.timestep()
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    /// Reference to the AdamW config.
    pub fn config(&self) -> &AdamWConfig {
        self.optimizer.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_zero_single_rank_matches_adamw() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let config = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        };

        // Set up identical params for both optimizers
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        let t1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let t2 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);

        let mut zero_params = HashMap::new();
        zero_params.insert(id1, t1.clone());
        zero_params.insert(id2, t2.clone());

        let mut adam_params = HashMap::new();
        adam_params.insert(id1, t1);
        adam_params.insert(id2, t2);

        let g1 = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let g2 = Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.4], &[2], &device);

        let mut grads = GradStore::new();
        grads.insert(id1, g1.clone());
        grads.insert(id2, g2.clone());

        let mut grads2 = GradStore::new();
        grads2.insert(id1, g1);
        grads2.insert(id2, g2);

        // ZeRO with world_size=1 (NoOp)
        let mut zero = ZeroStage1::<CpuRuntime>::new(config.clone(), comm, &[id1, id2]);
        zero.step(&client, &mut zero_params, &grads).unwrap();

        // Plain AdamW
        let mut adam = AdamW::<CpuRuntime>::new(config);
        adam.step(&client, &mut adam_params, &grads2).unwrap();

        // Results should be identical
        let z1: Vec<f32> = zero_params[&id1].to_vec();
        let a1: Vec<f32> = adam_params[&id1].to_vec();
        for (z, a) in z1.iter().zip(a1.iter()) {
            assert!((z - a).abs() < 1e-6, "mismatch: zero={z}, adam={a}");
        }

        let z2: Vec<f32> = zero_params[&id2].to_vec();
        let a2: Vec<f32> = adam_params[&id2].to_vec();
        for (z, a) in z2.iter().zip(a2.iter()) {
            assert!((z - a).abs() < 1e-6, "mismatch: zero={z}, adam={a}");
        }
    }

    #[test]
    fn test_zero_ownership_round_robin() {
        let comm = Arc::new(NoOpCommunicator); // rank=0, world_size=1

        let ids: Vec<TensorId> = (0..6).map(|_| TensorId::new()).collect();
        let zero = ZeroStage1::<CpuRuntime>::new(AdamWConfig::default(), comm, &ids);

        // With world_size=1, all params are owned
        assert_eq!(zero.owned_param_ids().len(), 6);
    }

    #[test]
    fn test_zero_step_updates_params() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id = TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[2], &device);
        let original: Vec<f32> = t.to_vec();

        let mut params = HashMap::new();
        params.insert(id, t);

        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(id, g);

        let config = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut zero = ZeroStage1::<CpuRuntime>::new(config, comm, &[id]);
        zero.step(&client, &mut params, &grads).unwrap();

        let updated: Vec<f32> = params[&id].to_vec();
        assert_ne!(updated, original, "params should change after step");
        assert_eq!(zero.timestep(), 1);
    }

    #[test]
    fn test_zero_set_lr() {
        let comm = Arc::new(NoOpCommunicator);
        let mut zero = ZeroStage1::<CpuRuntime>::new(AdamWConfig::default(), comm, &[]);
        zero.set_lr(0.01);
        assert_eq!(zero.config().lr, 0.01);
    }
}
