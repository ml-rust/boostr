//! Simple trainer for neural networks
//!
//! Integrates AdamW optimizer, gradient accumulation, gradient clipping,
//! and LR scheduling into a single training step abstraction.

use std::collections::HashMap;

use crate::error::Result;
use crate::optimizer::{AdamW, AdamWConfig, GradAccumulator, LrSchedule, clip_grad_norm};
use crate::trainer::config::{TrainingConfig, TrainingMetrics};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Simple single-device trainer
///
/// Manages the training loop plumbing:
/// - Gradient accumulation across micro-batches
/// - Gradient clipping by global norm
/// - Learning rate scheduling
/// - AdamW optimizer step
///
/// The user provides the forward/backward pass; the trainer handles
/// everything from gradients to parameter updates.
///
/// # Usage
///
/// ```ignore
/// let mut trainer = SimpleTrainer::new(config)?;
///
/// for micro_batch in data {
///     let loss = forward(micro_batch, &params);
///     let grads = backward(&loss, &client)?;
///
///     if let Some(metrics) = trainer.step(&client, &mut params, grads, loss_val)? {
///         println!("step {} loss={:.4} lr={:.6}", metrics.step, metrics.loss, metrics.lr);
///     }
/// }
/// ```
pub struct SimpleTrainer<R: Runtime> {
    optimizer: AdamW<R>,
    accumulator: GradAccumulator<R>,
    lr_schedule: Option<LrSchedule>,
    max_grad_norm: Option<f64>,
    global_step: u64,
    accumulated_loss: f64,
    loss_count: usize,
}

impl<R: Runtime<DType = DType>> SimpleTrainer<R> {
    pub fn new(config: TrainingConfig) -> Result<Self> {
        let optimizer = AdamW::new(AdamWConfig {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..AdamWConfig::default()
        });
        let accumulator = GradAccumulator::new(config.grad_accum_steps)?;

        Ok(Self {
            optimizer,
            accumulator,
            lr_schedule: None,
            max_grad_norm: config.max_grad_norm,
            global_step: 0,
            accumulated_loss: 0.0,
            loss_count: 0,
        })
    }

    pub fn with_lr_schedule(mut self, schedule: LrSchedule) -> Self {
        self.lr_schedule = Some(schedule);
        self
    }

    /// Process one micro-batch of gradients.
    ///
    /// Accumulates gradients. When enough micro-batches are accumulated,
    /// clips gradients, applies the optimizer step, and returns metrics.
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

        let averaged_grads = match self.accumulator.accumulate(client, grads)? {
            Some(g) => g,
            None => return Ok(None),
        };

        // Apply LR schedule
        if let Some(ref schedule) = self.lr_schedule {
            let lr = schedule.get_lr(self.global_step);
            self.optimizer.set_lr(lr);
        }

        // Gradient clipping
        let grad_norm = if let Some(max_norm) = self.max_grad_norm {
            let mut grads_mut = averaged_grads;
            let norm = clip_grad_norm(client, &mut grads_mut, max_norm)?;
            self.optimizer.step(client, params, &grads_mut)?;
            Some(norm)
        } else {
            self.optimizer.step(client, params, &averaged_grads)?;
            None
        };

        let avg_loss = self.accumulated_loss / self.loss_count as f64;
        self.accumulated_loss = 0.0;
        self.loss_count = 0;

        self.global_step += 1;

        Ok(Some(TrainingMetrics {
            step: self.global_step,
            loss: avg_loss,
            grad_norm,
            lr: self.optimizer.config().lr,
        }))
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn optimizer(&self) -> &AdamW<R> {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut AdamW<R> {
        &mut self.optimizer
    }
}
