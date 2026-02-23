//! ZeRO Stage 1 distributed trainer
//!
//! Integrates bucketed gradient allreduce with ZeRO Stage 1 optimizer state
//! partitioning. Gradients are allreduced normally, but the optimizer step
//! only updates owned params, then broadcasts them to all ranks.

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::grad_sync::all_reduce_grads;
use crate::distributed::zero::ZeroStage1;
use crate::distributed::zero_trainer_base::{ZeroTrainerBase, adamw_config_from_training};
use crate::error::Result;
use crate::trainer::config::{TrainingConfig, TrainingMetrics};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Distributed trainer with ZeRO Stage 1 optimizer state partitioning.
///
/// Each rank stores optimizer state (m, v) for only ~1/N of parameters.
/// Gradients are allreduced across ranks, then each rank updates only its
/// owned params and broadcasts the results.
pub struct ZeroTrainer<R: Runtime<DType = DType>> {
    base: ZeroTrainerBase<R, ZeroStage1<R>>,
}

impl<R: Runtime<DType = DType>> ZeroTrainer<R> {
    /// Create a new ZeRO Stage 1 trainer.
    pub fn new(
        config: TrainingConfig,
        comm: Arc<dyn Communicator>,
        param_ids: &[TensorId],
    ) -> Result<Self> {
        let adamw_config = adamw_config_from_training(&config);
        let zero_optimizer = ZeroStage1::new(adamw_config, comm.clone(), param_ids);
        let base = ZeroTrainerBase::new(&config, comm, zero_optimizer)?;
        Ok(Self { base })
    }

    crate::distributed::zero_trainer_base::impl_zero_trainer_common!();

    /// Process one micro-batch of gradients (consumed).
    ///
    /// 1. Allreduce gradients across ranks
    /// 2. Accumulate (if gradient accumulation is enabled)
    /// 3. Clip gradients (if configured)
    /// 4. ZeRO optimizer step (updates owned params + broadcasts)
    ///
    /// `grads` is consumed because accumulation merges them into internal
    /// state. Callers should not reuse `grads` after this call.
    ///
    /// Returns `None` if still accumulating, `Some(metrics)` after a full step.
    pub fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: GradStore<R>,
        loss_value: f64,
    ) -> Result<Option<TrainingMetrics>>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        // Stage 1: allreduce grads before accumulation
        let mut synced_grads = grads;
        all_reduce_grads(self.base.comm.as_ref(), client, &mut synced_grads)?;

        let mut averaged_grads = match self.base.prepare_step(client, synced_grads, loss_value)? {
            Some(g) => g,
            None => return Ok(None),
        };

        // ZeRO Stage 1 step (owned params + broadcast)
        self.base
            .zero_optimizer
            .step(client, params, &mut averaged_grads)?;

        Ok(Some(self.base.finish_step()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_zero_trainer_creation() {
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            grad_accum_steps: 1,
            max_grad_norm: Some(1.0),
        };
        let ids = vec![TensorId::new(), TensorId::new()];
        let trainer = ZeroTrainer::<CpuRuntime>::new(config, comm, &ids).unwrap();
        assert_eq!(trainer.global_step(), 0);
        assert_eq!(trainer.owned_param_ids().len(), 2);
    }

    #[test]
    fn test_zero_trainer_step() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-4,
            weight_decay: 0.0,
            grad_accum_steps: 1,
            max_grad_norm: None,
        };

        let param_id = TensorId::new();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(param_id, param);

        let mut trainer = ZeroTrainer::<CpuRuntime>::new(config, comm, &[param_id]).unwrap();

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(param_id, grad);

        let result = trainer.step(&client, &mut params, grads, 0.5).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().step, 1);
    }

    #[test]
    fn test_zero_trainer_grad_accumulation() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-4,
            weight_decay: 0.0,
            grad_accum_steps: 2,
            max_grad_norm: None,
        };

        let param_id = TensorId::new();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(param_id, param);

        let mut trainer = ZeroTrainer::<CpuRuntime>::new(config, comm, &[param_id]).unwrap();

        // First micro-batch: still accumulating
        let g1 = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let mut grads1 = GradStore::new();
        grads1.insert(param_id, g1);
        let result = trainer.step(&client, &mut params, grads1, 0.5).unwrap();
        assert!(result.is_none());

        // Second micro-batch: step fires
        let g2 = Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.4], &[2], &device);
        let mut grads2 = GradStore::new();
        grads2.insert(param_id, g2);
        let result = trainer.step(&client, &mut params, grads2, 0.6).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().step, 1);
    }

    #[test]
    fn test_zero_trainer_with_grad_clipping() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            grad_accum_steps: 1,
            max_grad_norm: Some(0.1),
        };

        let param_id = TensorId::new();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(param_id, param);

        let mut trainer = ZeroTrainer::<CpuRuntime>::new(config, comm, &[param_id]).unwrap();

        // Large gradient that should be clipped
        let g = Tensor::<CpuRuntime>::from_slice(&[100.0f32, 100.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(param_id, g);

        let result = trainer.step(&client, &mut params, grads, 1.0).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_zero_trainer_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ZeroTrainer<CpuRuntime>>();
    }
}
