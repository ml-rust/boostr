//! AdamW optimizer
//!
//! Implements decoupled weight decay regularization (Loshchilov & Hutter, 2019).
//! Uses numr tensor ops directly — works on any backend without GPU↔CPU transfers.

use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::traits::Optimizer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// AdamW configuration
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Per-parameter optimizer state
struct ParamState<R: Runtime> {
    m: Tensor<R>,
    v: Tensor<R>,
}

/// AdamW optimizer with decoupled weight decay
///
/// Maintains first moment (m) and second moment (v) estimates per parameter.
/// State is lazily initialized on first `step()` call for each parameter.
pub struct AdamW<R: Runtime> {
    config: AdamWConfig,
    state: HashMap<TensorId, ParamState<R>>,
    timestep: u64,
}

impl<R: Runtime<DType = DType>> AdamW<R> {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
            timestep: 0,
        }
    }

    /// Perform one optimization step.
    ///
    /// Updates all parameters in `params` using gradients from `grads`.
    /// Parameters without gradients are skipped.
    ///
    /// # Arguments
    /// * `client` - Runtime client for tensor ops
    /// * `params` - Mutable map of parameter ID → tensor
    /// * `grads` - Gradient store from `backward()`
    #[allow(clippy::type_complexity)]
    pub fn step<C>(
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
        self.timestep += 1;
        let t = self.timestep;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        // Bias correction factors
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        // Corrected learning rate: lr * sqrt(1 - beta2^t) / (1 - beta1^t)
        let step_size = lr * bc2.sqrt() / bc1;

        // Collect all param groups that have gradients
        let mut ids_with_grads: Vec<TensorId> = Vec::new();
        for &id in params.keys() {
            if grads.get(id).is_some() {
                // Lazy init state
                let param = params.get(&id).expect("iterating params.keys()");
                self.state.entry(id).or_insert_with(|| {
                    let m = Tensor::<R>::zeros(param.shape(), param.dtype(), param.device());
                    let v = Tensor::<R>::zeros(param.shape(), param.dtype(), param.device());
                    ParamState { m, v }
                });
                ids_with_grads.push(id);
            }
        }

        if ids_with_grads.is_empty() {
            return Ok(());
        }

        // Build groups for multi-tensor launch
        let groups: Vec<(&Tensor<R>, &Tensor<R>, &Tensor<R>, &Tensor<R>)> = ids_with_grads
            .iter()
            .map(|id| {
                let param = params.get(id).unwrap();
                let grad = grads.get(*id).unwrap();
                let state = self.state.get(id).unwrap();
                (param, grad, &state.m, &state.v)
            })
            .collect();

        let results =
            client.fused_multi_tensor_adamw(&groups, lr, beta1, beta2, eps, wd, step_size)?;

        // Write back results
        for (id, (new_param, new_m, new_v)) in ids_with_grads.iter().zip(results) {
            let state_mut = self.state.get_mut(id).unwrap();
            state_mut.m = new_m;
            state_mut.v = new_v;
            params.insert(*id, new_param);
        }

        Ok(())
    }

    pub fn timestep(&self) -> u64 {
        self.timestep
    }

    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }

    pub fn reset(&mut self) {
        self.state.clear();
        self.timestep = 0;
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

impl<R: Runtime<DType = DType>> Optimizer<R> for AdamW<R> {
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
        AdamW::step(self, client, params, grads)
    }

    fn set_lr(&mut self, lr: f64) {
        AdamW::set_lr(self, lr);
    }

    fn lr(&self) -> f64 {
        self.config.lr
    }

    fn reset(&mut self) {
        AdamW::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, backward};
    use numr::autograd::{var_matmul, var_mean};
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_adamw_default_config() {
        let config = AdamWConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_adamw_single_step() {
        let (client, device) = cpu_setup();

        // Create parameter tensor and Var — use Var's id as the canonical key
        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let w = Var::new(w_tensor, true);
        let w_id = w.id();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], &device);
        let x_var = Var::new(x, false);

        // Forward: loss = mean(w @ x)
        let out = var_matmul(&w, &x_var, &client).unwrap();
        let loss = var_mean(&out, &[0, 1], false, &client).unwrap();

        // Backward
        let grads = backward(&loss, &client).unwrap();

        // Optimizer step — insert param with the Var's id
        // Note: clone() creates a new TensorId, so we must use the Var's id as key
        // and the original tensor data
        let mut params = HashMap::new();
        let w_data = w.tensor().clone();
        params.insert(w_id, w_data);

        let config = AdamWConfig::default();
        let mut opt = AdamW::<CpuRuntime>::new(config);

        opt.step(&client, &mut params, &grads).unwrap();

        assert_eq!(opt.timestep(), 1);

        // Parameter should have changed
        let updated = params.get(&w_id).unwrap();
        let updated_data = updated.to_vec::<f32>();
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_ne!(updated_data, original, "params should change after step");
    }

    #[test]
    fn test_adamw_multiple_steps_decrease_loss() {
        let (client, device) = cpu_setup();

        // Simple optimization: minimize ||w - target||^2
        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);

        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        // Use a stable ID for the parameter across all iterations
        let w_id = w_init.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_init);

        let config = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.0, // no decay for this test
            ..Default::default()
        };
        let mut opt = AdamW::<CpuRuntime>::new(config);

        let mut first_loss = 0.0f64;
        let mut last_loss = 0.0f64;

        for i in 0..20 {
            // Wrap current param as Var with the SAME id so grads map back
            let w_tensor = params.get(&w_id).unwrap().clone();
            let w = Var::with_id(w_tensor, w_id, true);
            let t = Var::new(target.clone(), false);

            // loss = mean((w - target)^2)
            let diff = numr::autograd::var_sub(&w, &t, &client).unwrap();
            let sq = numr::autograd::var_mul(&diff, &diff, &client).unwrap();
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
            last_loss < first_loss * 0.1,
            "loss should decrease significantly: first={} last={}",
            first_loss,
            last_loss
        );

        // After 20 steps, w should be close to target
        let final_w = params.get(&w_id).unwrap().to_vec::<f32>();
        assert!(
            (final_w[0] - 1.0).abs() < 0.3,
            "w[0] should approach 1.0, got {}",
            final_w[0]
        );
    }

    #[test]
    fn test_adamw_weight_decay() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[2], &device);
        let w_id = w_tensor.id();

        // Create a zero gradient — weight decay should still shrink params
        let mut grads = GradStore::new();
        let zero_grad = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);
        grads.insert(w_id, zero_grad);

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let config = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut opt = AdamW::<CpuRuntime>::new(config);

        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        // With zero grad, only weight decay applies: param = param - lr * wd * param
        // = 5.0 - 0.1 * 0.1 * 5.0 = 5.0 - 0.05 = 4.95
        // (plus a tiny adam update from eps, but close to 4.95)
        assert!(
            updated[0] < 5.0,
            "weight decay should shrink params, got {}",
            updated[0]
        );
    }

    #[test]
    fn test_adamw_reset() {
        let opt: AdamW<CpuRuntime> = AdamW::new(AdamWConfig::default());
        assert_eq!(opt.timestep(), 0);
    }

    #[test]
    fn test_adamw_set_lr() {
        let mut opt: AdamW<CpuRuntime> = AdamW::new(AdamWConfig::default());
        opt.set_lr(0.01);
        assert_eq!(opt.config().lr, 0.01);
    }

    #[test]
    fn test_adamw_skips_missing_grads() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor.clone());

        // Empty grad store — no grads for any param
        let grads = GradStore::new();

        let mut opt = AdamW::<CpuRuntime>::new(AdamWConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        // Param should be unchanged (no grad = no update)
        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert_eq!(updated, vec![1.0, 2.0]);
    }
}
