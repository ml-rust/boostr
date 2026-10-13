//! Shared base for ZeRO Stage 1/2/3 optimizers.
//!
//! `ZeroOptimizerBase` holds the common fields (AdamW, communicator, ownership
//! maps) and provides the shared constructor + helper methods that all three
//! stages delegate to.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::optimizer::{AdamW, AdamWConfig};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};

use crate::ops::FusedOptimizerOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

use crate::distributed::comm_utils::broadcast_tensor;

/// Common interface for all ZeRO optimizer stages (Stage 1, 2, and 3).
///
/// This trait enables the shared trainer base to work generically across all
/// stages without knowing which stage is used. Users typically don't implement
/// this themselves; it is automatically implemented by [`ZeroStage1`],
/// [`ZeroStage2`], and [`ZeroStage3`].
///
/// [`ZeroStage1`]: super::zero::ZeroStage1
/// [`ZeroStage2`]: super::zero2::ZeroStage2
/// [`ZeroStage3`]: super::zero3::ZeroStage3
pub trait ZeroOptimizer<R: Runtime> {
    /// The set of parameter IDs this rank owns.
    fn owned_param_ids(&self) -> &HashSet<TensorId>;
    /// Current optimizer timestep (number of completed optimizer steps).
    fn timestep(&self) -> u64;
    /// Set learning rate.
    fn set_lr(&mut self, lr: f64);
    /// Reference to the AdamW config.
    fn config(&self) -> &AdamWConfig;

    /// Gather full parameters from all ranks (Stage 3 only).
    ///
    /// Default is a no-op for Stage 1 and Stage 2 which keep full params.
    fn gather_full_params(
        &self,
        _params: &mut HashMap<TensorId, Tensor<R>>,
        _device: &R::Device,
    ) -> Result<()> {
        Ok(())
    }

    /// Release non-owned parameters (Stage 3 only).
    ///
    /// Default is a no-op for Stage 1 and Stage 2 which keep full params.
    fn release_params(&self, _params: &mut HashMap<TensorId, Tensor<R>>) {}
}

/// Implements the [`ZeroOptimizer`] trait for a type wrapping [`ZeroOptimizerBase`].
///
/// Expects the implementing type to have a `base: ZeroOptimizerBase<R>` field.
macro_rules! impl_zero_optimizer {
    ($ty:ident) => {
        impl<R: numr::runtime::Runtime<DType = numr::dtype::DType>>
            $crate::distributed::zero_base::ZeroOptimizer<R> for $ty<R>
        {
            fn owned_param_ids(&self) -> &std::collections::HashSet<numr::tensor::TensorId> {
                &self.base.owned_params
            }

            fn timestep(&self) -> u64 {
                self.base.optimizer.timestep()
            }

            fn set_lr(&mut self, lr: f64) {
                self.base.optimizer.set_lr(lr);
            }

            fn config(&self) -> &$crate::optimizer::AdamWConfig {
                self.base.optimizer.config()
            }
        }
    };
}

pub(crate) use impl_zero_optimizer;

/// Shared base struct for ZeRO Stage 1/2/3 optimizers.
///
/// Holds AdamW optimizer, communicator, and the round-robin parameter
/// ownership mapping that is identical across all three stages.
pub(crate) struct ZeroOptimizerBase<R: Runtime> {
    pub(crate) optimizer: AdamW<R>,
    pub(crate) comm: Arc<dyn Communicator>,
    pub(crate) world_size: usize,
    pub(crate) owned_params: HashSet<TensorId>,
    pub(crate) param_owners: Vec<(TensorId, usize)>,
}

impl<R: Runtime<DType = DType>> ZeroOptimizerBase<R> {
    /// Create a new base with round-robin parameter assignment.
    pub(crate) fn new(
        config: AdamWConfig,
        comm: Arc<dyn Communicator>,
        param_ids: &[TensorId],
    ) -> Self {
        let rank = comm.rank();
        let world_size = comm.world_size();

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

    /// Filter a GradStore to only contain gradients for owned params.
    pub(crate) fn filter_to_owned(&self, grads: &GradStore<R>) -> GradStore<R> {
        let mut owned = GradStore::new();
        for &id in &self.owned_params {
            if let Some(g) = grads.get(id) {
                owned.insert(id, g.clone());
            }
        }
        owned
    }

    /// Move owned params out of the full params map into a separate map.
    pub(crate) fn extract_owned_params(
        &self,
        params: &mut HashMap<TensorId, Tensor<R>>,
    ) -> HashMap<TensorId, Tensor<R>> {
        let mut owned_map = HashMap::new();
        for &id in &self.owned_params {
            if let Some(t) = params.remove(&id) {
                owned_map.insert(id, t);
            }
        }
        owned_map
    }

    /// Move owned params back into the full params map.
    pub(crate) fn restore_owned_params(
        &self,
        owned_map: HashMap<TensorId, Tensor<R>>,
        params: &mut HashMap<TensorId, Tensor<R>>,
    ) {
        for (id, tensor) in owned_map {
            params.insert(id, tensor);
        }
    }

    /// Broadcast updated params from each owner to all ranks, then sync.
    pub(crate) fn broadcast_owned_params(
        &self,
        params: &HashMap<TensorId, Tensor<R>>,
        stage_name: &str,
    ) -> Result<()> {
        for &(id, owner) in &self.param_owners {
            if let Some(tensor) = params.get(&id) {
                broadcast_tensor(self.comm.as_ref(), tensor, owner)?;
            }
        }

        self.comm.sync().map_err(|e| Error::DistributedError {
            reason: format!("sync after {stage_name} broadcast failed: {e}"),
        })?;

        Ok(())
    }

    /// Run AdamW on owned params only (extract, step, restore).
    pub(crate) fn step_owned<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + FusedOptimizerOps<R>,
    {
        let mut owned_map = self.extract_owned_params(params);
        self.optimizer.step(client, &mut owned_map, grads)?;
        self.restore_owned_params(owned_map, params);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_base_ownership_round_robin() {
        let comm = Arc::new(NoOpCommunicator); // rank=0, world_size=1
        let ids: Vec<TensorId> = (0..6).map(|_| TensorId::new()).collect();
        let base = ZeroOptimizerBase::<CpuRuntime>::new(AdamWConfig::default(), comm, &ids);
        // With world_size=1, all params are owned
        assert_eq!(base.owned_params.len(), 6);
    }

    #[test]
    fn test_base_set_lr() {
        let comm = Arc::new(NoOpCommunicator);
        let mut base = ZeroOptimizerBase::<CpuRuntime>::new(AdamWConfig::default(), comm, &[]);
        base.optimizer.set_lr(0.01);
        assert_eq!(base.optimizer.config().lr, 0.01);
    }
}
