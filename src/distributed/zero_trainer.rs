//! ZeRO Stage 1 distributed trainer
//!
//! Integrates bucketed gradient allreduce with ZeRO Stage 1 optimizer state
//! partitioning. Gradients are allreduced normally, but the optimizer step
//! only updates owned params, then broadcasts them to all ranks.

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::grad_sync::{all_reduce_grads, broadcast_params};
use crate::distributed::zero::ZeroStage1;
use crate::error::Result;
use crate::optimizer::{AdamWConfig, GradAccumulator, LrSchedule, clip_grad_norm};
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
///
/// # Usage
///
/// ```ignore
/// let comm = Arc::new(NcclCommunicator::new(...)?);
/// let param_ids: Vec<TensorId> = params.keys().copied().collect();
/// let mut trainer = ZeroTrainer::new(config, comm, &param_ids)?;
///
/// trainer.broadcast_params(&params)?;
///
/// for micro_batch in local_data {
///     let loss = forward(micro_batch, &params);
///     let grads = backward(&loss, &client)?;
///     if let Some(metrics) = trainer.step(&client, &mut params, grads, loss_val)? {
///         println!("step {} loss={:.4}", metrics.step, metrics.loss);
///     }
/// }
/// ```
pub struct ZeroTrainer<R: Runtime> {
    zero_optimizer: ZeroStage1<R>,
    accumulator: GradAccumulator<R>,
    lr_schedule: Option<LrSchedule>,
    max_grad_norm: Option<f64>,
    global_step: u64,
    accumulated_loss: f64,
    loss_count: usize,
    comm: Arc<dyn Communicator>,
}

impl<R: Runtime<DType = DType>> ZeroTrainer<R> {
    /// Create a new ZeRO Stage 1 trainer.
    ///
    /// # Arguments
    /// * `config` - Training configuration
    /// * `comm` - Communicator for collective operations
    /// * `param_ids` - All parameter IDs (must be the same on all ranks)
    pub fn new(
        config: TrainingConfig,
        comm: Arc<dyn Communicator>,
        param_ids: &[TensorId],
    ) -> Result<Self> {
        let adamw_config = AdamWConfig {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..AdamWConfig::default()
        };
        let zero_optimizer = ZeroStage1::new(adamw_config, comm.clone(), param_ids);
        let accumulator = GradAccumulator::new(config.grad_accum_steps)?;

        Ok(Self {
            zero_optimizer,
            accumulator,
            lr_schedule: None,
            max_grad_norm: config.max_grad_norm,
            global_step: 0,
            accumulated_loss: 0.0,
            loss_count: 0,
            comm,
        })
    }

    /// Set a learning rate schedule.
    pub fn with_lr_schedule(mut self, schedule: LrSchedule) -> Self {
        self.lr_schedule = Some(schedule);
        self
    }

    /// Broadcast parameters from rank 0 to all other ranks.
    pub fn broadcast_params(&self, params: &HashMap<TensorId, Tensor<R>>) -> Result<()> {
        broadcast_params(self.comm.as_ref(), params, 0)
    }

    /// Process one micro-batch of gradients.
    ///
    /// 1. Allreduce gradients across ranks
    /// 2. Accumulate (if gradient accumulation is enabled)
    /// 3. Clip gradients (if configured)
    /// 4. ZeRO optimizer step (updates owned params + broadcasts)
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
        self.accumulated_loss += loss_value;
        self.loss_count += 1;

        // Allreduce gradients across ranks
        let mut synced_grads = grads;
        all_reduce_grads(self.comm.as_ref(), client, &mut synced_grads)?;

        // Accumulate micro-batches
        let averaged_grads = match self.accumulator.accumulate(client, synced_grads)? {
            Some(g) => g,
            None => return Ok(None),
        };

        // Apply LR schedule
        if let Some(ref schedule) = self.lr_schedule {
            let lr = schedule.get_lr(self.global_step);
            self.zero_optimizer.set_lr(lr);
        }

        // Gradient clipping then ZeRO optimizer step
        if let Some(max_norm) = self.max_grad_norm {
            let mut grads_mut = averaged_grads;
            clip_grad_norm(client, &mut grads_mut, max_norm)?;
            self.zero_optimizer.step(client, params, &grads_mut)?;
        } else {
            self.zero_optimizer.step(client, params, &averaged_grads)?;
        }

        let avg_loss = self.accumulated_loss / self.loss_count as f64;
        self.accumulated_loss = 0.0;
        self.loss_count = 0;

        self.global_step += 1;

        Ok(Some(TrainingMetrics {
            step: self.global_step,
            loss: avg_loss,
            grad_norm: None,
            lr: self.zero_optimizer.config().lr,
        }))
    }

    /// Current global step count.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Reference to the underlying communicator.
    pub fn communicator(&self) -> &dyn Communicator {
        self.comm.as_ref()
    }

    /// The set of parameter IDs this rank owns optimizer state for.
    pub fn owned_param_ids(&self) -> &std::collections::HashSet<TensorId> {
        self.zero_optimizer.owned_param_ids()
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
