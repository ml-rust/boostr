//! The LAMB/LARS `Optimizer` trait bridge.
//!
//! The per-parameter update math (the trust-ratio loop and its L2 norm)
//! lives in `super::apply`.

use super::types::Lamb;
use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::traits::Optimizer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> Optimizer<R> for Lamb<R> {
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
            + TypeConversionOps<R>
            + FusedOptimizerOps<R>,
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

        self.apply_updates(
            client, params, grads, param_ids, beta1, beta2, lr, eps, wd, bc1, bc2,
        )
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
    use super::super::types::LambConfig;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_lamb_converges() {
        let (client, device) = cpu_setup();

        let target =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let w_init =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device).unwrap();
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

        let target =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let w_init =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device).unwrap();
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
        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[100.0f32, 100.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.001f32, 0.001], &[2], &device).unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let grads = GradStore::new();
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert_eq!(updated, vec![1.0, 2.0]);
    }
}
