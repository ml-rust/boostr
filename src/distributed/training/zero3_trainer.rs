//! ZeRO Stage 3 (FSDP) distributed trainer
//!
//! Integrates gradient handling with ZeRO Stage 3 full parameter sharding.
//! The caller is responsible for calling `gather_full_params` before forward/backward
//! and `release_params` after the step to manage the parameter lifecycle.

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::zero_trainer_base::{ZeroTrainerBase, adamw_config_from_training};
use crate::distributed::zero3::ZeroStage3;
use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::trainer::config::{TrainingConfig, TrainingMetrics};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Distributed trainer with ZeRO Stage 3 (FSDP) full parameter sharding.
///
/// At rest, each rank holds only ~1/N of parameters. The training loop is:
///
/// 1. `gather_full_params()` — reconstruct full params for forward/backward
/// 2. Forward + backward → produce grads
/// 3. `step()` — accumulate, clip, allreduce, optimizer on owned params
/// 4. `release_params()` — drop non-owned params, return to sharded state
pub struct Zero3Trainer<R: Runtime<DType = DType>> {
    base: ZeroTrainerBase<R, ZeroStage3<R>>,
}

impl<R: Runtime<DType = DType>> Zero3Trainer<R> {
    /// Create a new ZeRO Stage 3 trainer.
    ///
    /// `params` must contain ALL parameters so that shape/dtype metadata can
    /// be captured for `gather_full_params` receive buffer allocation.
    pub fn new(
        config: TrainingConfig,
        comm: Arc<dyn Communicator>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Result<Self> {
        let adamw_config = adamw_config_from_training(&config);
        let zero3 = ZeroStage3::new(adamw_config, comm.clone(), params);
        let base = ZeroTrainerBase::new(&config, comm, zero3)?;
        Ok(Self { base })
    }

    crate::distributed::zero_trainer_base::impl_zero_trainer_common!();

    /// Gather full parameters from all ranks for forward/backward.
    pub fn gather_full_params(
        &self,
        params: &mut HashMap<TensorId, Tensor<R>>,
        device: &R::Device,
    ) -> Result<()> {
        self.base.zero_optimizer.gather_full_params(params, device)
    }

    /// Release non-owned parameters after backward+step.
    pub fn release_params(&self, params: &mut HashMap<TensorId, Tensor<R>>) {
        self.base.zero_optimizer.release_params(params);
    }

    /// Process one micro-batch of gradients (consumed).
    ///
    /// Assumes full params are currently gathered (caller called `gather_full_params`).
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
        C: RuntimeClient<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + FusedOptimizerOps<R>,
    {
        let mut averaged_grads = match self.base.prepare_step(client, grads, loss_value)? {
            Some(g) => g,
            None => return Ok(None),
        };

        // ZeRO Stage 3 step (allreduce + filter + adamw, no broadcast)
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
    fn test_zero3_trainer_creation() {
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            grad_accum_steps: 1,
            max_grad_norm: Some(1.0),
        };
        let (_, device) = cpu_setup();
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        let t1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
        let t2 = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);
        let mut params = HashMap::new();
        params.insert(id1, t1);
        params.insert(id2, t2);
        let trainer = Zero3Trainer::<CpuRuntime>::new(config, comm, &params).unwrap();
        assert_eq!(trainer.global_step(), 0);
        assert_eq!(trainer.owned_param_ids().len(), 2);
    }

    #[test]
    fn test_zero3_trainer_step() {
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

        let mut trainer = Zero3Trainer::<CpuRuntime>::new(config, comm, &params).unwrap();

        trainer.gather_full_params(&mut params, &device).unwrap();

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(param_id, grad);

        let result = trainer.step(&client, &mut params, grads, 0.5).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().step, 1);

        trainer.release_params(&mut params);
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_zero3_trainer_grad_accumulation() {
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

        let mut trainer = Zero3Trainer::<CpuRuntime>::new(config, comm, &params).unwrap();

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
    fn test_zero3_trainer_with_grad_clipping() {
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

        let mut trainer = Zero3Trainer::<CpuRuntime>::new(config, comm, &params).unwrap();

        // Large gradient that should be clipped
        let g = Tensor::<CpuRuntime>::from_slice(&[100.0f32, 100.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(param_id, g);

        let result = trainer.step(&client, &mut params, grads, 1.0).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_zero3_trainer_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Zero3Trainer<CpuRuntime>>();
    }
}
