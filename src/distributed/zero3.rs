//! ZeRO Stage 3 (FSDP) optimizer — shards optimizer state + gradients + parameters
//!
//! Each rank holds only ~1/N of parameters at rest. Full parameters are
//! reconstructed on-demand via all-gather (`gather_full_params`) for
//! forward/backward, then released after use (`release_params`).
//! The optimizer step only runs on owned params without broadcasting —
//! params stay sharded until the next `gather_full_params` call.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::distributed::comm_utils::broadcast_tensor;
use crate::distributed::grad_sync::all_reduce_grads;
use crate::distributed::zero_base::{ZeroOptimizer, ZeroOptimizerBase};
use crate::error::{Error, Result};
use crate::optimizer::AdamWConfig;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Metadata needed to allocate a receive buffer for a non-owned parameter.
#[derive(Clone, Debug)]
struct ParamMeta {
    shape: Vec<usize>,
    dtype: DType,
}

/// ZeRO Stage 3: full parameter, gradient, and optimizer state partitioning.
///
/// At rest, each rank only holds its owned ~1/N of parameters. Before
/// forward/backward, call `gather_full_params` to reconstruct all params
/// via broadcast from each owner. After backward+step, call `release_params`
/// to drop non-owned params and return to the sharded state.
pub struct ZeroStage3<R: Runtime> {
    base: ZeroOptimizerBase<R>,
    param_meta: HashMap<TensorId, ParamMeta>,
}

impl<R: Runtime<DType = DType>> ZeroStage3<R> {
    /// Create a new ZeRO Stage 3 optimizer.
    ///
    /// Parameters are assigned round-robin by sorted TensorId to ranks.
    /// Each rank will only store parameters, optimizer state, and gradients
    /// for its owned subset.
    ///
    /// `params` must contain ALL parameters so that shape/dtype metadata can
    /// be captured. Non-owner ranks need this metadata to allocate receive
    /// buffers during `gather_full_params`.
    pub fn new(
        config: AdamWConfig,
        comm: Arc<dyn Communicator>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Self {
        let param_ids: Vec<TensorId> = params.keys().copied().collect();
        let base = ZeroOptimizerBase::new(config, comm, &param_ids);

        let mut param_meta = HashMap::with_capacity(params.len());
        for (&id, tensor) in params {
            param_meta.insert(
                id,
                ParamMeta {
                    shape: tensor.shape().to_vec(),
                    dtype: tensor.dtype(),
                },
            );
        }

        Self { base, param_meta }
    }

    /// Gather full parameters from all ranks via broadcast.
    ///
    /// Each owner broadcasts its params to all other ranks, reconstructing
    /// the full parameter map. Non-owner ranks allocate a receive buffer
    /// using stored shape/dtype metadata, then participate in the collective
    /// broadcast. After this call, `params` contains every parameter.
    ///
    /// Call this before forward/backward passes.
    ///
    /// With `world_size=1`, this is a no-op (all params are already present).
    pub fn gather_full_params(
        &self,
        params: &mut HashMap<TensorId, Tensor<R>>,
        device: &R::Device,
    ) -> Result<()> {
        if self.base.world_size <= 1 {
            return Ok(());
        }

        for &(id, owner) in &self.base.param_owners {
            use std::collections::hash_map::Entry;
            if let Entry::Vacant(e) = params.entry(id) {
                let meta = self
                    .param_meta
                    .get(&id)
                    .ok_or_else(|| Error::DistributedError {
                        reason: format!(
                            "missing metadata for param {id:?} — was it in the \
                             original params passed to ZeroStage3::new?"
                        ),
                    })?;
                let buf = Tensor::<R>::zeros(&meta.shape, meta.dtype, device);
                e.insert(buf);
            }

            let tensor = params.get(&id).expect("just ensured it exists");
            broadcast_tensor(self.base.comm.as_ref(), tensor, owner)?;
        }

        self.base.comm.sync().map_err(|e| Error::DistributedError {
            reason: format!("sync after ZeRO Stage 3 gather failed: {e}"),
        })?;

        Ok(())
    }

    /// Release non-owned parameters, returning to the sharded state.
    ///
    /// Removes all non-owned parameter tensors from the map, freeing their
    /// memory. Call this after backward+step to minimize memory usage.
    ///
    /// With `world_size=1`, this is a no-op (all params are owned).
    pub fn release_params(&self, params: &mut HashMap<TensorId, Tensor<R>>) {
        if self.base.world_size <= 1 {
            return;
        }

        params.retain(|id, _| self.base.owned_params.contains(id));
    }

    /// Perform one ZeRO Stage 3 optimizer step.
    ///
    /// Assumes full params are currently gathered (from `gather_full_params`).
    ///
    /// 1. Allreduce all gradients (average across ranks)
    /// 2. Filter grads to only owned params
    /// 3. Run AdamW on owned params only
    /// 4. Does NOT broadcast — params stay sharded until next `gather_full_params`
    pub fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &mut GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R>,
    {
        if self.base.world_size <= 1 {
            return self.base.optimizer.step(client, params, grads);
        }

        // Step 1: Allreduce gradients
        all_reduce_grads(self.base.comm.as_ref(), client, grads)?;

        // Step 2: Filter grads to owned params only, free non-owned gradient memory
        let owned_grads = self.base.filter_to_owned(grads);
        grads.clear();

        // Step 3: Run AdamW on owned params only (no broadcast)
        self.base.step_owned(client, params, &owned_grads)?;

        Ok(())
    }

    /// All param IDs with their owning rank, in deterministic order.
    pub fn param_owners(&self) -> &[(TensorId, usize)] {
        &self.base.param_owners
    }
}

// Use the macro for the 4 common methods, then manually impl the Stage 3 overrides.
// Unfortunately Rust doesn't support partial macro trait impls, so we do it manually
// but still eliminate the boilerplate for the common methods.
impl<R: Runtime<DType = DType>> ZeroOptimizer<R> for ZeroStage3<R> {
    fn owned_param_ids(&self) -> &HashSet<TensorId> {
        &self.base.owned_params
    }

    fn timestep(&self) -> u64 {
        self.base.optimizer.timestep()
    }

    fn set_lr(&mut self, lr: f64) {
        self.base.optimizer.set_lr(lr);
    }

    fn config(&self) -> &AdamWConfig {
        self.base.optimizer.config()
    }

    fn gather_full_params(
        &self,
        params: &mut HashMap<TensorId, Tensor<R>>,
        device: &R::Device,
    ) -> Result<()> {
        ZeroStage3::gather_full_params(self, params, device)
    }

    fn release_params(&self, params: &mut HashMap<TensorId, Tensor<R>>) {
        ZeroStage3::release_params(self, params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_zero3_single_rank_matches_adamw() {
        use crate::optimizer::AdamW;

        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let config = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        };

        let id1 = TensorId::new();
        let id2 = TensorId::new();

        let t1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let t2 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);

        let mut zero3_params = HashMap::new();
        zero3_params.insert(id1, t1.clone());
        zero3_params.insert(id2, t2.clone());

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

        // ZeRO Stage 3 with world_size=1 should match plain AdamW
        let mut zero3 = ZeroStage3::<CpuRuntime>::new(config.clone(), comm, &zero3_params);
        zero3.step(&client, &mut zero3_params, &mut grads).unwrap();

        let mut adam = AdamW::<CpuRuntime>::new(config);
        adam.step(&client, &mut adam_params, &grads2).unwrap();

        // Results should be identical
        let z1: Vec<f32> = zero3_params[&id1].to_vec();
        let a1: Vec<f32> = adam_params[&id1].to_vec();
        for (z, a) in z1.iter().zip(a1.iter()) {
            assert!((z - a).abs() < 1e-6, "mismatch: zero3={z}, adam={a}");
        }
    }

    #[test]
    fn test_zero3_gather_release_lifecycle() {
        let (_, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id1 = TensorId::new();
        let id2 = TensorId::new();

        let t1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let t2 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);

        let mut params = HashMap::new();
        params.insert(id1, t1);
        params.insert(id2, t2);

        let zero3 = ZeroStage3::<CpuRuntime>::new(AdamWConfig::default(), comm, &params);

        assert_eq!(params.len(), 2);
        zero3.gather_full_params(&mut params, &device).unwrap();
        assert_eq!(params.len(), 2);
        zero3.release_params(&mut params);
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_zero3_step_updates_owned_params() {
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
        let mut zero3 = ZeroStage3::<CpuRuntime>::new(config, comm, &params);
        zero3.step(&client, &mut params, &mut grads).unwrap();

        let updated: Vec<f32> = params[&id].to_vec();
        assert_ne!(updated, original, "params should change after step");
        assert_eq!(zero3.timestep(), 1);
    }

    #[test]
    fn test_zero3_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ZeroStage3<CpuRuntime>>();
    }
}
