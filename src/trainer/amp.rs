//! Mixed precision (AMP) trainer
//!
//! Wraps `SimpleTrainer` with automatic mixed precision: forward/backward in BF16/FP16
//! with FP32 master weights for numerical stability. Achieves ~2x throughput on modern GPUs.

use std::collections::HashMap;

use crate::error::Result;
use crate::optimizer::grad_scaler::{GradScaler, UnscaleResult};
use crate::trainer::config::{MixedPrecisionConfig, TrainingConfig, TrainingMetrics};
use crate::trainer::simple::SimpleTrainer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Mixed precision trainer
///
/// Maintains FP32 master weights and provides BF16/FP16 compute copies.
/// The workflow per step:
///
/// 1. `compute_params()` — cast master params to compute dtype
/// 2. User runs forward/backward with compute params
/// 3. `step()` — unscale grads (FP16 only), cast to FP32, run optimizer on master params
///
/// # Example
///
/// ```ignore
/// let config = TrainingConfig::default();
/// let amp_config = MixedPrecisionConfig::bf16();
/// let mut trainer = AmpTrainer::new(config, amp_config, initial_params)?;
///
/// for batch in data {
///     let compute_params = trainer.compute_params(&client)?;
///     let loss = forward(batch, &compute_params);
///     let grads = backward(&loss, &client)?;
///     if let Some(metrics) = trainer.step(&client, grads, loss_val)? {
///         println!("step {} loss={:.4}", metrics.step, metrics.loss);
///     }
/// }
/// ```
pub struct AmpTrainer<R: Runtime<DType = DType>> {
    trainer: SimpleTrainer<R>,
    master_params: HashMap<TensorId, Tensor<R>>,
    compute_dtype: DType,
    grad_scaler: Option<GradScaler>,
}

impl<R: Runtime<DType = DType>> AmpTrainer<R> {
    /// Create a new mixed precision trainer.
    ///
    /// # Arguments
    /// * `config` - Training configuration (lr, weight decay, etc.)
    /// * `amp_config` - Mixed precision configuration (compute dtype, loss scaling)
    /// * `initial_params` - Initial model parameters (any dtype, will be cast to FP32 as master)
    pub fn new<C>(
        config: TrainingConfig,
        amp_config: MixedPrecisionConfig,
        initial_params: HashMap<TensorId, Tensor<R>>,
        client: &C,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + TypeConversionOps<R>,
    {
        let trainer = SimpleTrainer::new(config)?;

        // Create FP32 master copies
        let mut master_params = HashMap::with_capacity(initial_params.len());
        for (id, param) in initial_params {
            let master = if param.dtype() == DType::F32 {
                param
            } else {
                client.cast(&param, DType::F32)?
            };
            master_params.insert(id, master);
        }

        let grad_scaler = amp_config.loss_scale.to_grad_scaler()?;

        Ok(Self {
            trainer,
            master_params,
            compute_dtype: amp_config.compute_dtype,
            grad_scaler,
        })
    }

    /// Get compute-dtype copies of all parameters for forward/backward.
    ///
    /// Cast master params (FP32) to the compute dtype (BF16/FP16).
    pub fn compute_params<C>(&self, client: &C) -> Result<HashMap<TensorId, Tensor<R>>>
    where
        C: RuntimeClient<R> + TypeConversionOps<R>,
    {
        let mut compute = HashMap::with_capacity(self.master_params.len());
        for (&id, master) in &self.master_params {
            let param = if master.dtype() == self.compute_dtype {
                master.clone()
            } else {
                client.cast(master, self.compute_dtype)?
            };
            compute.insert(id, param);
        }
        Ok(compute)
    }

    /// Process one micro-batch of gradients with mixed precision.
    ///
    /// Handles:
    /// 1. FP16 loss unscaling and overflow detection (if applicable)
    /// 2. Casting gradients to FP32
    /// 3. Delegating to SimpleTrainer for accumulation, clipping, and optimizer step
    ///
    /// Returns `None` if still accumulating or if an overflow was detected (FP16).
    /// Returns `Some(metrics)` after a full optimizer step.
    pub fn step<C>(
        &mut self,
        client: &C,
        grads: GradStore<R>,
        loss_value: f64,
    ) -> Result<Option<TrainingMetrics>>
    where
        C: RuntimeClient<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + TypeConversionOps<R>,
    {
        // Step 1: If FP16 with grad scaler, unscale and check for overflow
        let grads = if let Some(ref scaler) = self.grad_scaler {
            match scaler.unscale_grads(client, grads)? {
                UnscaleResult::Ok(unscaled) => unscaled,
                UnscaleResult::Overflow => {
                    if let Some(ref mut s) = self.grad_scaler {
                        s.update_scale(true);
                    }
                    return Ok(None);
                }
            }
        } else {
            grads
        };

        // Step 2: Cast gradients to FP32 for optimizer
        let mut fp32_grads = GradStore::new();
        let ids: Vec<TensorId> = grads.keys().copied().collect();
        for id in ids {
            let grad = grads
                .get(id)
                .ok_or_else(|| crate::error::Error::TrainingError {
                    reason: format!("missing gradient for tensor {id:?}"),
                })?;
            let fp32_grad = if grad.dtype() == DType::F32 {
                grad.clone()
            } else {
                client.cast(grad, DType::F32)?
            };
            fp32_grads.insert(id, fp32_grad);
        }

        // Step 3: Run optimizer step on master params
        let result = self
            .trainer
            .step(client, &mut self.master_params, fp32_grads, loss_value)?;

        // Step 4: Update grad scaler if present (no overflow)
        if let Some(ref mut scaler) = self.grad_scaler {
            scaler.update_scale(false);
        }

        Ok(result)
    }

    /// Get the current loss scale factor (for FP16).
    ///
    /// Returns 1.0 if no grad scaler is active (BF16 mode).
    pub fn loss_scale(&self) -> f64 {
        self.grad_scaler.as_ref().map_or(1.0, |s| s.scale())
    }

    /// Scale a loss value before backward (for FP16).
    ///
    /// Returns the loss unchanged for BF16 mode.
    pub fn scale_loss(&self, loss: f64) -> f64 {
        self.grad_scaler
            .as_ref()
            .map_or(loss, |s| s.scale_loss(loss))
    }

    /// Get a reference to the master parameters (FP32).
    pub fn master_params(&self) -> &HashMap<TensorId, Tensor<R>> {
        &self.master_params
    }

    /// Get the compute dtype.
    pub fn compute_dtype(&self) -> DType {
        self.compute_dtype
    }

    /// Get the current global step count.
    pub fn global_step(&self) -> u64 {
        self.trainer.global_step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use crate::trainer::config::LossScaleStrategy;
    use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
    use numr::ops::TypeConversionOps;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_amp_trainer_f64_compute_converges() {
        let (client, device) = cpu_setup();

        // Use F64 as compute dtype to test on CPU (BF16 needs f16 feature)
        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        let w_id = w_init.id();

        let mut initial_params = HashMap::new();
        initial_params.insert(w_id, w_init);

        let config = TrainingConfig::default()
            .with_lr(0.1)
            .with_weight_decay(0.0)
            .with_max_grad_norm(None);
        // Use F64 compute with F32 master to test the cast pathway
        let amp_config = MixedPrecisionConfig {
            compute_dtype: DType::F64,
            master_dtype: DType::F32,
            loss_scale: LossScaleStrategy::None,
        };

        let mut trainer = AmpTrainer::new(config, amp_config, initial_params, &client).unwrap();

        let mut first_loss = 0.0f64;
        let mut last_loss = 0.0f64;

        for i in 0..20 {
            let compute_params = trainer.compute_params(&client).unwrap();
            let w_tensor = compute_params.get(&w_id).unwrap().clone();
            assert_eq!(w_tensor.dtype(), DType::F64);

            let w = Var::with_id(w_tensor, w_id, true);
            let t_f64 = client.cast(&target, DType::F64).unwrap();
            let t = Var::new(t_f64, false);

            let diff = var_sub(&w, &t, &client).unwrap();
            let sq = var_mul(&diff, &diff, &client).unwrap();
            let loss = var_mean(&sq, &[], false, &client).unwrap();

            let loss_val = loss.tensor().to_vec::<f64>()[0];
            if i == 0 {
                first_loss = loss_val;
            }
            last_loss = loss_val;

            let grads = backward(&loss, &client).unwrap();
            trainer.step(&client, grads, loss_val).unwrap();
        }

        assert!(
            last_loss < first_loss * 0.1,
            "loss should decrease: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_amp_trainer_loss_scale() {
        let (client, device) = cpu_setup();

        let w = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
        let mut params = HashMap::new();
        params.insert(w.id(), w);

        // BF16: no scaler
        let trainer = AmpTrainer::<CpuRuntime>::new(
            TrainingConfig::default(),
            MixedPrecisionConfig::bf16(),
            params.clone(),
            &client,
        )
        .unwrap();
        assert_eq!(trainer.loss_scale(), 1.0);
        assert_eq!(trainer.scale_loss(2.5), 2.5);

        // FP16: has scaler
        let trainer = AmpTrainer::<CpuRuntime>::new(
            TrainingConfig::default(),
            MixedPrecisionConfig::fp16(),
            params,
            &client,
        )
        .unwrap();
        assert_eq!(trainer.loss_scale(), 65536.0);
        assert_eq!(trainer.scale_loss(1.0), 65536.0);
    }

    #[test]
    fn test_amp_trainer_master_params_are_f32() {
        let (client, device) = cpu_setup();

        let w = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        let id = w.id();
        params.insert(id, w);

        let trainer = AmpTrainer::<CpuRuntime>::new(
            TrainingConfig::default(),
            MixedPrecisionConfig::bf16(),
            params,
            &client,
        )
        .unwrap();

        assert_eq!(
            trainer.master_params().get(&id).unwrap().dtype(),
            DType::F32
        );
    }
}
