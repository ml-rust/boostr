//! LAMB optimizer (Layer-wise Adaptive Moments for Batch training)
//!
//! You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes", 2020.
//! Layer-wise adaptive scaling enables stable training at very large batch sizes (32K+).
//! Used by Google for BERT pre-training and applicable to frontier-scale LLM training.
//!
//! Also supports LARS mode (Layer-wise Adaptive Rate Scaling, You et al., 2017)
//! by setting `use_adam = false`, which uses SGD-style momentum instead of Adam moments.

use crate::error::Result;
use crate::optimizer::traits::Optimizer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// LAMB / LARS configuration
#[derive(Debug, Clone)]
pub struct LambConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// Trust ratio clipping. If set, clamps the trust ratio to [0, max_trust_ratio].
    pub max_trust_ratio: Option<f64>,
    /// If true, use Adam-style moments (LAMB). If false, use SGD momentum (LARS).
    pub use_adam: bool,
}

impl Default for LambConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
            max_trust_ratio: Some(10.0),
            use_adam: true,
        }
    }
}

impl LambConfig {
    /// LARS configuration (SGD momentum with layer-wise scaling)
    pub fn lars() -> Self {
        Self {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.0,
            eps: 1e-6,
            weight_decay: 1e-4,
            max_trust_ratio: Some(10.0),
            use_adam: false,
        }
    }
}

struct LambState<R: Runtime> {
    m: Tensor<R>,
    v: Tensor<R>,
}

/// LAMB optimizer with layer-wise adaptive trust ratios
///
/// Computes Adam (or SGD momentum) updates per parameter, then scales each
/// layer's update by `||param|| / ||update||` (the "trust ratio"). This
/// normalization keeps gradient magnitudes consistent across layers,
/// enabling stable training at batch sizes of 32K+.
pub struct Lamb<R: Runtime> {
    config: LambConfig,
    state: HashMap<TensorId, LambState<R>>,
    timestep: u64,
}

impl<R: Runtime<DType = DType>> Lamb<R> {
    pub fn new(config: LambConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
            timestep: 0,
        }
    }

    pub fn config(&self) -> &LambConfig {
        &self.config
    }

    pub fn timestep(&self) -> u64 {
        self.timestep
    }
}

/// Compute L2 norm of a tensor as f64, device-native via reduction ops.
fn tensor_l2_norm<R, C>(client: &C, t: &Tensor<R>) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R>,
{
    let sq = client.mul(t, t)?;
    let sum_sq = client.sum(&sq, &[], false)?;
    let val: f32 = sum_sq.item()?;
    Ok((val as f64).sqrt())
}

impl<R: Runtime<DType = DType>> Optimizer<R> for Lamb<R> {
    fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        self.timestep += 1;
        let t = self.timestep;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = if self.config.use_adam {
            1.0 - beta2.powi(t as i32)
        } else {
            1.0
        };

        let param_ids: Vec<TensorId> = params.keys().copied().collect();

        for id in param_ids {
            let grad = match grads.get(id) {
                Some(g) => g,
                None => continue,
            };

            let param = params.get(&id).expect("id collected from params.keys()");

            // Lazy init
            self.state.entry(id).or_insert_with(|| {
                let m = Tensor::<R>::zeros(param.shape(), DType::F32, param.device());
                let v = Tensor::<R>::zeros(param.shape(), DType::F32, param.device());
                LambState { m, v }
            });

            let state = self.state.get(&id).unwrap();

            // Update moments
            // m = beta1 * m + (1 - beta1) * grad
            let m_scaled = client.mul_scalar(&state.m, beta1)?;
            let g_scaled = client.mul_scalar(grad, 1.0 - beta1)?;
            let new_m = client.add(&m_scaled, &g_scaled)?;

            let new_v = if self.config.use_adam {
                // v = beta2 * v + (1 - beta2) * grad^2
                let v_scaled = client.mul_scalar(&state.v, beta2)?;
                let g_sq = client.mul(grad, grad)?;
                let g_sq_scaled = client.mul_scalar(&g_sq, 1.0 - beta2)?;
                client.add(&v_scaled, &g_sq_scaled)?
            } else {
                state.v.clone()
            };

            // Bias-corrected estimates
            let m_hat = client.mul_scalar(&new_m, 1.0 / bc1)?;

            // Compute raw update
            let update = if self.config.use_adam {
                let v_hat = client.mul_scalar(&new_v, 1.0 / bc2)?;
                let v_sqrt = client.sqrt(&v_hat)?;
                let denom = client.add_scalar(&v_sqrt, eps)?;
                client.div(&m_hat, &denom)?
            } else {
                m_hat
            };

            // Add weight decay to update: update = update + wd * param
            let update = if wd > 0.0 {
                let decay_term = client.mul_scalar(param, wd)?;
                client.add(&update, &decay_term)?
            } else {
                update
            };

            // Compute trust ratio: phi(||param||) / ||update||
            let param_norm = tensor_l2_norm(client, param)?;
            let update_norm = tensor_l2_norm(client, &update)?;

            let trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
                let ratio = param_norm / update_norm;
                match self.config.max_trust_ratio {
                    Some(max) => ratio.min(max),
                    None => ratio,
                }
            } else {
                1.0
            };

            // param = param - lr * trust_ratio * update
            let effective_lr = lr * trust_ratio;
            let scaled_update = client.mul_scalar(&update, effective_lr)?;
            let new_param = client.sub(param, &scaled_update)?;

            let state_mut = self.state.get_mut(&id).unwrap();
            state_mut.m = new_m;
            state_mut.v = new_v;
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
        self.state.clear();
        self.timestep = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_lamb_default_config() {
        let config = LambConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert!(config.use_adam);
        assert_eq!(config.max_trust_ratio, Some(10.0));
    }

    #[test]
    fn test_lars_config() {
        let config = LambConfig::lars();
        assert_eq!(config.lr, 0.1);
        assert!(!config.use_adam);
    }

    #[test]
    fn test_lamb_converges() {
        let (client, device) = cpu_setup();

        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        let w_id = w_init.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_init);

        let mut opt = Lamb::<CpuRuntime>::new(LambConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        });

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
            last_loss < first_loss * 0.1,
            "LAMB should converge: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_lars_converges() {
        let (client, device) = cpu_setup();

        let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
        let w_id = w_init.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_init);

        let mut opt = Lamb::<CpuRuntime>::new(LambConfig {
            weight_decay: 0.0,
            ..LambConfig::lars()
        });

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
            last_loss < first_loss * 0.1,
            "LARS should converge: first={first_loss} last={last_loss}"
        );
    }

    #[test]
    fn test_lamb_trust_ratio_clamped() {
        let (client, device) = cpu_setup();

        // Large param, tiny gradient → trust ratio would be huge without clamping
        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[100.0f32, 100.0], &[2], &device);
        let w_id = w_tensor.id();

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.001f32, 0.001], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(w_id, grad);

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let mut opt = Lamb::<CpuRuntime>::new(LambConfig {
            lr: 0.01,
            weight_decay: 0.0,
            max_trust_ratio: Some(10.0),
            ..Default::default()
        });

        opt.step(&client, &mut params, &grads).unwrap();

        // Should not explode
        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert!(
            updated[0].is_finite(),
            "update should be finite: {}",
            updated[0]
        );
        assert!(
            (updated[0] - 100.0).abs() < 1.0,
            "clamped trust ratio should limit step size: {}",
            updated[0]
        );
    }

    #[test]
    fn test_lamb_skips_missing_grads() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let grads = GradStore::new();
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert_eq!(updated, vec![1.0, 2.0]);
    }

    #[test]
    fn test_lamb_reset() {
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.reset();
        assert_eq!(opt.timestep(), 0);
        assert!(opt.state.is_empty());
    }
}
