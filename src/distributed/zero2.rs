//! ZeRO Stage 2 optimizer — shards optimizer state + gradients across data-parallel ranks
//!
//! Extends Stage 1 by explicitly filtering gradients to only owned parameters
//! before the optimizer step, freeing non-owned gradient memory. With per-param
//! sharding this is nearly identical to Stage 1, but the explicit gradient
//! filtering guarantees memory savings proportional to 1/N for both optimizer
//! state AND gradient storage.

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::grad_sync::all_reduce_grads;
use crate::distributed::zero_base::ZeroOptimizerBase;
use crate::error::Result;
use crate::optimizer::AdamWConfig;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// ZeRO Stage 2: optimizer state + gradient partitioning across data-parallel ranks.
///
/// Each rank only stores optimizer state (m, v) AND gradients for its owned
/// subset of parameters. After allreduce, non-owned gradients are dropped
/// before the optimizer step, then updated params are broadcast from owners.
///
/// With `world_size=1`, this degenerates to plain `AdamW` with no communication.
pub struct ZeroStage2<R: Runtime> {
    base: ZeroOptimizerBase<R>,
}

impl<R: Runtime<DType = DType>> ZeroStage2<R> {
    /// Create a new ZeRO Stage 2 optimizer.
    ///
    /// Parameters are assigned round-robin by sorted TensorId to ranks.
    /// Each rank will only store optimizer state and gradients for its owned params.
    pub fn new(config: AdamWConfig, comm: Arc<dyn Communicator>, param_ids: &[TensorId]) -> Self {
        Self {
            base: ZeroOptimizerBase::new(config, comm, param_ids),
        }
    }

    /// Perform one ZeRO Stage 2 optimizer step.
    ///
    /// 1. Allreduce all gradients (average across ranks)
    /// 2. Filter grads to only owned params (Stage 2 memory saving)
    /// 3. Run AdamW on owned params only
    /// 4. Broadcast updated params from each owner to all ranks
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

        // Step 1: Allreduce gradients across all ranks
        all_reduce_grads(self.base.comm.as_ref(), client, grads)?;

        // Step 2: Filter grads to only owned params, free non-owned gradient memory
        let owned_grads = self.base.filter_to_owned(grads);
        grads.clear();

        // Step 3: Run AdamW on owned params only
        self.base.step_owned(client, params, &owned_grads)?;

        // Step 4: Broadcast updated params from each owner
        self.base.broadcast_owned_params(params, "ZeRO Stage 2")?;

        Ok(())
    }
}

crate::distributed::zero_base::impl_zero_optimizer!(ZeroStage2);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::zero_base::ZeroOptimizer;
    use crate::optimizer::AdamW;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_zero2_single_rank_matches_adamw() {
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

        let mut zero2_params = HashMap::new();
        zero2_params.insert(id1, t1.clone());
        zero2_params.insert(id2, t2.clone());

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

        // ZeRO Stage 2 with world_size=1 should match plain AdamW
        let mut zero2 = ZeroStage2::<CpuRuntime>::new(config.clone(), comm, &[id1, id2]);
        zero2.step(&client, &mut zero2_params, &mut grads).unwrap();

        let mut adam = AdamW::<CpuRuntime>::new(config);
        adam.step(&client, &mut adam_params, &grads2).unwrap();

        // Results should be identical
        let z1: Vec<f32> = zero2_params[&id1].to_vec();
        let a1: Vec<f32> = adam_params[&id1].to_vec();
        for (z, a) in z1.iter().zip(a1.iter()) {
            assert!((z - a).abs() < 1e-6, "mismatch: zero2={z}, adam={a}");
        }
    }

    #[test]
    fn test_zero2_step_updates_params() {
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
        let mut zero2 = ZeroStage2::<CpuRuntime>::new(config, comm, &[id]);
        zero2.step(&client, &mut params, &mut grads).unwrap();

        let updated: Vec<f32> = params[&id].to_vec();
        assert_ne!(updated, original, "params should change after step");
        assert_eq!(zero2.timestep(), 1);
    }

    #[test]
    fn test_zero2_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ZeroStage2<CpuRuntime>>();
    }
}
