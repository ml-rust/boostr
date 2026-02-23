//! Shared base for ZeRO Stage 1/2/3 trainers.
//!
//! `ZeroTrainerBase` holds the common training loop state (accumulator,
//! LR schedule, grad clipping, loss tracking) and provides `prepare_step`
//! / `finish_step` helpers that all three trainer stages delegate to.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::distributed::grad_sync::broadcast_params;
use crate::distributed::zero_base::ZeroOptimizer;
use crate::error::Result;
use crate::optimizer::{AdamWConfig, GradAccumulator, LrSchedule, clip_grad_norm};
use crate::trainer::config::{TrainingConfig, TrainingMetrics};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Shared base struct for ZeRO trainers.
pub(crate) struct ZeroTrainerBase<R: Runtime<DType = DType>, Z: ZeroOptimizer<R>> {
    pub(crate) zero_optimizer: Z,
    pub(crate) accumulator: GradAccumulator<R>,
    pub(crate) lr_schedule: Option<LrSchedule>,
    pub(crate) max_grad_norm: Option<f64>,
    pub(crate) global_step: u64,
    pub(crate) accumulated_loss: f64,
    pub(crate) loss_count: usize,
    pub(crate) comm: Arc<dyn Communicator>,
}

impl<R: Runtime<DType = DType>, Z: ZeroOptimizer<R>> ZeroTrainerBase<R, Z> {
    /// Create a new trainer base.
    pub(crate) fn new(
        config: &TrainingConfig,
        comm: Arc<dyn Communicator>,
        zero_optimizer: Z,
    ) -> Result<Self> {
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
    pub(crate) fn set_lr_schedule(&mut self, schedule: LrSchedule) {
        self.lr_schedule = Some(schedule);
    }

    /// Broadcast parameters from rank 0 to all other ranks.
    pub(crate) fn broadcast_params(&self, params: &HashMap<TensorId, Tensor<R>>) -> Result<()> {
        broadcast_params(self.comm.as_ref(), params, 0)
    }

    /// Accumulate loss + grads, apply LR schedule + clipping.
    ///
    /// Returns `None` if still accumulating, `Some(grads)` ready for optimizer.
    pub(crate) fn prepare_step<C>(
        &mut self,
        client: &C,
        grads: GradStore<R>,
        loss_value: f64,
    ) -> Result<Option<GradStore<R>>>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        self.accumulated_loss += loss_value;
        self.loss_count += 1;

        let mut averaged_grads = match self.accumulator.accumulate(client, grads)? {
            Some(g) => g,
            None => return Ok(None),
        };

        // Apply LR schedule
        if let Some(ref schedule) = self.lr_schedule {
            let lr = schedule.get_lr(self.global_step);
            self.zero_optimizer.set_lr(lr);
        }

        // Gradient clipping
        if let Some(max_norm) = self.max_grad_norm {
            clip_grad_norm(client, &mut averaged_grads, max_norm)?;
        }

        Ok(Some(averaged_grads))
    }

    /// Compute metrics and advance step counter. Call after optimizer step.
    pub(crate) fn finish_step(&mut self) -> TrainingMetrics {
        let avg_loss = self.accumulated_loss / self.loss_count as f64;
        self.accumulated_loss = 0.0;
        self.loss_count = 0;
        self.global_step += 1;

        TrainingMetrics {
            step: self.global_step,
            loss: avg_loss,
            grad_norm: None,
            lr: self.zero_optimizer.config().lr,
        }
    }

    /// Current global step count.
    pub(crate) fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Reference to the underlying communicator.
    pub(crate) fn communicator(&self) -> &dyn Communicator {
        self.comm.as_ref()
    }

    /// The set of parameter IDs this rank owns optimizer state for.
    pub(crate) fn owned_param_ids(&self) -> &HashSet<TensorId> {
        self.zero_optimizer.owned_param_ids()
    }
}

/// Helper to build AdamWConfig from TrainingConfig.
pub(crate) fn adamw_config_from_training(config: &TrainingConfig) -> AdamWConfig {
    AdamWConfig {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..AdamWConfig::default()
    }
}

/// Implements the common delegation methods shared by all ZeRO trainer wrappers.
///
/// Generates `with_lr_schedule`, `broadcast_params`, `global_step`,
/// `communicator`, and `owned_param_ids` methods that simply forward to the
/// inner `ZeroTrainerBase`.
macro_rules! impl_zero_trainer_common {
    () => {
        /// Set a learning rate schedule.
        pub fn with_lr_schedule(mut self, schedule: $crate::optimizer::LrSchedule) -> Self {
            self.base.set_lr_schedule(schedule);
            self
        }

        /// Broadcast parameters from rank 0 to all other ranks.
        pub fn broadcast_params(
            &self,
            params: &std::collections::HashMap<numr::tensor::TensorId, numr::tensor::Tensor<R>>,
        ) -> $crate::error::Result<()> {
            self.base.broadcast_params(params)
        }

        /// Current global step count.
        pub fn global_step(&self) -> u64 {
            self.base.global_step()
        }

        /// Reference to the underlying communicator.
        pub fn communicator(&self) -> &dyn numr::runtime::Communicator {
            self.base.communicator()
        }

        /// The set of parameter IDs this rank owns optimizer state for.
        pub fn owned_param_ids(&self) -> &std::collections::HashSet<numr::tensor::TensorId> {
            self.base.owned_param_ids()
        }
    };
}

pub(crate) use impl_zero_trainer_common;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::zero::ZeroStage1;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_trainer_rejects_zero_accum_steps() {
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            grad_accum_steps: 0,
            max_grad_norm: None,
        };
        let adamw_config = adamw_config_from_training(&config);
        let optimizer = ZeroStage1::<CpuRuntime>::new(adamw_config, comm.clone(), &[]);
        let result = ZeroTrainerBase::new(&config, comm, optimizer);
        assert!(result.is_err(), "grad_accum_steps=0 should be rejected");
    }

    #[test]
    fn test_adamw_config_from_training() {
        let config = TrainingConfig {
            learning_rate: 0.042,
            weight_decay: 0.1,
            grad_accum_steps: 1,
            max_grad_norm: None,
        };
        let adamw = adamw_config_from_training(&config);
        assert_eq!(adamw.lr, 0.042);
        assert_eq!(adamw.weight_decay, 0.1);
    }
}
