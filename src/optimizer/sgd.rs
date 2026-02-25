//! SGD optimizer with momentum
//!
//! Implements stochastic gradient descent with optional momentum and weight decay.
//! Follows PyTorch's SGD semantics with Nesterov momentum support.

use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::traits::Optimizer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// SGD configuration
#[derive(Debug, Clone)]
pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub dampening: f64,
    pub nesterov: bool,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
        }
    }
}

/// SGD optimizer with optional momentum
///
/// When `momentum > 0`, maintains a velocity buffer per parameter.
/// Supports Nesterov momentum for improved convergence.
///
/// Update rules (following PyTorch):
/// - L2 weight decay: `grad = grad + weight_decay * param`
/// - Momentum: `buf = momentum * buf + (1 - dampening) * grad`
/// - Nesterov: `update = grad + momentum * buf`
/// - Standard: `update = buf`
/// - Parameter: `param = param - lr * update`
pub struct Sgd<R: Runtime> {
    config: SgdConfig,
    velocity: HashMap<TensorId, Tensor<R>>,
}

impl<R: Runtime<DType = DType>> Sgd<R> {
    pub fn new(config: SgdConfig) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
        }
    }

    pub fn config(&self) -> &SgdConfig {
        &self.config
    }
}

impl<R: Runtime<DType = DType>> Optimizer<R> for Sgd<R> {
    fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + FusedOptimizerOps<R>,
    {
        let lr = self.config.lr;
        let momentum = self.config.momentum;
        let wd = self.config.weight_decay;
        let dampening = self.config.dampening;
        let nesterov = self.config.nesterov;

        let param_ids: Vec<TensorId> = params.keys().copied().collect();

        for id in param_ids {
            let grad = match grads.get(id) {
                Some(g) => g,
                None => continue,
            };

            let param = params.get(&id).expect("id collected from params.keys()");

            let momentum_buf = self.velocity.get(&id);

            let (new_param, new_buf) = client.fused_sgd_step(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            )?;

            if momentum > 0.0 {
                self.velocity.insert(id, new_buf);
            }
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
        self.velocity.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_sgd_default_config() {
        let config = SgdConfig::default();
        assert_eq!(config.lr, 0.01);
        assert_eq!(config.momentum, 0.0);
        assert_eq!(config.weight_decay, 0.0);
        assert_eq!(config.dampening, 0.0);
        assert!(!config.nesterov);
    }

    #[test]
    fn test_sgd_vanilla_step() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let w_id = w_tensor.id();

        // grad = [0.1, 0.2, 0.3, 0.4]
        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2], &device);
        let mut grads = GradStore::new();
        grads.insert(w_id, grad);

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let config = SgdConfig {
            lr: 0.1,
            ..Default::default()
        };
        let mut opt = Sgd::<CpuRuntime>::new(config);

        opt.step(&client, &mut params, &grads).unwrap();

        // param = param - lr * grad = [1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03, 4.0 - 0.04]
        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert!((updated[0] - 0.99).abs() < 1e-6);
        assert!((updated[1] - 1.98).abs() < 1e-6);
        assert!((updated[2] - 2.97).abs() < 1e-6);
        assert!((updated[3] - 3.96).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum_converges() {
        let (client, device) = cpu_setup();

        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        let w_id = w_init.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_init);

        let config = SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        };
        let mut opt = Sgd::<CpuRuntime>::new(config);

        let mut first_loss = 0.0f64;
        let mut last_loss = 0.0f64;

        for i in 0..50 {
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
    fn test_sgd_nesterov() {
        let (client, device) = cpu_setup();

        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        let w_id = w_init.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_init);

        let config = SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            nesterov: true,
            ..Default::default()
        };
        let mut opt = Sgd::<CpuRuntime>::new(config);

        let mut first_loss = 0.0f64;
        let mut last_loss = 0.0f64;

        for i in 0..50 {
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
            "nesterov should converge: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_sgd_weight_decay() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[2], &device);
        let w_id = w_tensor.id();

        let zero_grad = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);
        let mut grads = GradStore::new();
        grads.insert(w_id, zero_grad);

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let config = SgdConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut opt = Sgd::<CpuRuntime>::new(config);

        opt.step(&client, &mut params, &grads).unwrap();

        // grad = 0 + 0.1 * 5.0 = 0.5, param = 5.0 - 0.1 * 0.5 = 4.95
        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert!(
            (updated[0] - 4.95).abs() < 1e-5,
            "weight decay: got {}",
            updated[0]
        );
    }

    #[test]
    fn test_sgd_skips_missing_grads() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let grads = GradStore::new();
        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert_eq!(updated, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sgd_reset() {
        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig {
            momentum: 0.9,
            ..Default::default()
        });
        opt.reset();
        assert!(opt.velocity.is_empty());
    }

    #[test]
    fn test_sgd_set_lr() {
        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }
}
