//! AdaGrad optimizer
//!
//! Adaptive gradient algorithm (Duchi et al., 2011). Adapts learning rates
//! per-parameter based on accumulated squared gradients. Particularly effective
//! for sparse gradients (e.g., embedding layers).

use crate::error::Result;
use crate::optimizer::traits::Optimizer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// AdaGrad configuration
#[derive(Debug, Clone)]
pub struct AdaGradConfig {
    pub lr: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// Initial accumulator value. Non-zero values help stabilize early steps.
    pub initial_accumulator_value: f64,
}

impl Default for AdaGradConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            eps: 1e-10,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
        }
    }
}

/// AdaGrad optimizer
///
/// Maintains a sum of squared gradients per parameter. The effective learning
/// rate decreases over time as the accumulator grows, which naturally anneals
/// the step size without requiring an explicit schedule.
///
/// Update rule:
/// - `accum = accum + grad^2`
/// - `param = param - lr * grad / (sqrt(accum) + eps)`
pub struct AdaGrad<R: Runtime> {
    config: AdaGradConfig,
    accumulators: HashMap<TensorId, Tensor<R>>,
}

impl<R: Runtime<DType = DType>> AdaGrad<R> {
    pub fn new(config: AdaGradConfig) -> Self {
        Self {
            config,
            accumulators: HashMap::new(),
        }
    }

    pub fn config(&self) -> &AdaGradConfig {
        &self.config
    }
}

impl<R: Runtime<DType = DType>> Optimizer<R> for AdaGrad<R> {
    fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        let lr = self.config.lr;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;
        let init_val = self.config.initial_accumulator_value;

        let param_ids: Vec<TensorId> = params.keys().copied().collect();

        for id in param_ids {
            let grad = match grads.get(id) {
                Some(g) => g,
                None => continue,
            };

            let param = params.get(&id).expect("id collected from params.keys()");

            // L2 weight decay
            let grad = if wd > 0.0 {
                let decay_term = client.mul_scalar(param, wd)?;
                client.add(grad, &decay_term)?
            } else {
                grad.clone()
            };

            // Lazy init accumulator
            if let std::collections::hash_map::Entry::Vacant(e) = self.accumulators.entry(id) {
                let acc = if init_val == 0.0 {
                    Tensor::<R>::zeros(param.shape(), DType::F32, param.device())
                } else {
                    let z = Tensor::<R>::zeros(param.shape(), DType::F32, param.device());
                    client.add_scalar(&z, init_val)?
                };
                e.insert(acc);
            }

            let acc = self
                .accumulators
                .get(&id)
                .expect("accumulator just initialized");

            // accum = accum + grad^2
            let grad_sq = client.mul(&grad, &grad)?;
            let new_acc = client.add(acc, &grad_sq)?;

            // param = param - lr * grad / (sqrt(accum) + eps)
            let acc_sqrt = client.sqrt(&new_acc)?;
            let denom = client.add_scalar(&acc_sqrt, eps)?;
            let update = client.div(&grad, &denom)?;
            let scaled = client.mul_scalar(&update, lr)?;
            let new_param = client.sub(param, &scaled)?;

            self.accumulators.insert(id, new_acc);
            params.insert(id, new_param);
        }

        Ok(())
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    fn lr(&self) -> f64 {
        self.config.lr
    }

    fn reset(&mut self) {
        self.accumulators.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_adagrad_default_config() {
        let config = AdaGradConfig::default();
        assert_eq!(config.lr, 0.01);
        assert_eq!(config.eps, 1e-10);
        assert_eq!(config.weight_decay, 0.0);
        assert_eq!(config.initial_accumulator_value, 0.0);
    }

    #[test]
    fn test_adagrad_single_step() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let w_id = w_tensor.id();

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(w_id, grad);

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig {
            lr: 0.1,
            ..Default::default()
        });

        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        // After first step: accum = grad^2, update = lr * grad / sqrt(grad^2) = lr * sign(grad)
        // So each element decreases by lr = 0.1
        assert!((updated[0] - 0.9).abs() < 1e-4);
        assert!((updated[1] - 1.9).abs() < 1e-4);
    }

    #[test]
    fn test_adagrad_converges() {
        let (client, device) = cpu_setup();

        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        let w_id = w_init.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_init);

        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig {
            lr: 0.5,
            ..Default::default()
        });

        let mut first_loss = 0.0f64;
        let mut last_loss = 0.0f64;

        for i in 0..100 {
            let w_tensor = params.get(&w_id).unwrap().clone();
            let w = Var::with_id(w_tensor, w_id, true);
            let t = Var::new(target.clone(), false);

            let diff = var_sub(&w, &t, &client).unwrap();
            let sq = var_mul(&diff, &diff, &client).unwrap();
            let loss = var_mean(&sq, &[0, 1], false, &client).unwrap();

            let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
            if i == 0 {
                first_loss = loss_val;
            }
            last_loss = loss_val;

            let grads = backward(&loss, &client).unwrap();
            opt.step(&client, &mut params, &grads).unwrap();
        }

        assert!(
            last_loss < first_loss * 0.01,
            "loss should decrease: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_adagrad_lr_decreases_over_time() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig {
            lr: 1.0,
            ..Default::default()
        });

        // Same gradient each step — effective LR should decrease
        let mut prev_delta = f64::MAX;
        for _ in 0..5 {
            let before = params.get(&w_id).unwrap().to_vec::<f32>()[0] as f64;

            let grad = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
            let mut grads = GradStore::new();
            grads.insert(w_id, grad);

            opt.step(&client, &mut params, &grads).unwrap();

            let after = params.get(&w_id).unwrap().to_vec::<f32>()[0] as f64;
            let delta = (before - after).abs();
            assert!(
                delta < prev_delta,
                "effective step size should decrease: {delta} >= {prev_delta}"
            );
            prev_delta = delta;
        }
    }

    #[test]
    fn test_adagrad_weight_decay() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[2], &device);
        let w_id = w_tensor.id();

        let zero_grad = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);
        let mut grads = GradStore::new();
        grads.insert(w_id, zero_grad);

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        });

        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert!(
            updated[0] < 5.0,
            "weight decay should shrink params, got {}",
            updated[0]
        );
    }

    #[test]
    fn test_adagrad_skips_missing_grads() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let grads = GradStore::new();
        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert_eq!(updated, vec![1.0, 2.0]);
    }

    #[test]
    fn test_adagrad_reset() {
        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig::default());
        opt.reset();
        assert!(opt.accumulators.is_empty());
    }
}
