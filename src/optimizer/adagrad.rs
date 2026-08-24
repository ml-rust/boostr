//! AdaGrad optimizer
//!
//! Adaptive gradient algorithm (Duchi et al., 2011). Adapts learning rates
//! per-parameter based on accumulated squared gradients. Particularly effective
//! for sparse gradients (e.g., embedding layers).

use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::precision::optimizer_state_dtype;
use crate::optimizer::traits::Optimizer;
use crate::optimizer::{init_master, widen_grad, write_back};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

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

/// Per-parameter optimizer state
struct ParamState<R: Runtime> {
    /// Sum of squared gradients. Always held at the optimizer state dtype: the
    /// accumulator grows monotonically, so a narrow one saturates and the
    /// effective step size collapses.
    acc: Tensor<R>,
    /// F32 master copy of the parameter, held ONLY when the parameter's own
    /// dtype is narrower than F32 (BF16/F16/FP8).
    ///
    /// The update runs against the master and a cast of the master is written
    /// back into the caller's `params` map, so the model keeps computing in its
    /// own dtype while the update arithmetic stays F32. For an F32 or F64
    /// parameter this is `None`: no copy, no extra allocation, and the numbers
    /// are bit-identical to a build without master weights.
    master: Option<Tensor<R>>,
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
///
/// For a parameter narrower than F32 (BF16/F16/FP8) the optimizer also holds an
/// F32 master copy and keeps the accumulator at F32: at fine-tuning learning
/// rates the update is smaller than BF16's resolution, so updating the narrow
/// parameter directly rounds every step away and the model never trains. See
/// [`crate::optimizer::precision`].
///
/// Optimizer state is not persisted by this type — a resumed run rebuilds the
/// master copies from the checkpointed parameters on its first step.
pub struct AdaGrad<R: Runtime> {
    config: AdaGradConfig,
    state: HashMap<TensorId, ParamState<R>>,
}

impl<R: Runtime<DType = DType>> AdaGrad<R> {
    pub fn new(config: AdaGradConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
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
        C: RuntimeClient<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + TypeConversionOps<R>
            + FusedOptimizerOps<R>,
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

            let param = match params.get(&id) {
                Some(p) => p,
                None => continue,
            };

            let param_dtype = param.dtype();
            let state_dtype = optimizer_state_dtype(param_dtype);

            // Entry rather than contains_key + insert: both the accumulator and
            // the master copy need fallible calls, so `or_insert_with` cannot
            // build them.
            let state = match self.state.entry(id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let zeros = Tensor::<R>::try_zeros(param.shape(), state_dtype, param.device())?;
                    let acc = if init_val == 0.0 {
                        zeros
                    } else {
                        client.add_scalar(&zeros, init_val)?
                    };
                    let master = init_master(client, param, state_dtype)?;
                    entry.insert(ParamState { acc, master })
                }
            };

            let arith_param = state.master.as_ref().unwrap_or(param);

            let widened_grad = widen_grad(client, grad, state_dtype)?;
            let arith_grad = widened_grad.as_ref().unwrap_or(grad);

            let (new_param, new_acc) =
                client.fused_adagrad_step(arith_param, arith_grad, &state.acc, lr, eps, wd)?;

            state.acc = new_acc;

            let updated = write_back(client, state.master.as_mut(), new_param, param_dtype)?;
            params.insert(id, updated);
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let grad = Tensor::<CpuRuntime>::try_from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();
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

        let target =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)
                .unwrap();
        let w_init =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device)
                .unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[5.0f32], &[1], &device).unwrap();
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

            let grad = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32], &[1], &device).unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[5.0f32, 5.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let zero_grad = Tensor::<CpuRuntime>::try_zeros(&[2], DType::F32, &device).unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
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
        assert!(opt.state.is_empty());
    }

    #[test]
    fn test_adagrad_set_lr() {
        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }

    /// Plain f32 AdaGrad, mirroring the kernel arithmetic element-wise.
    #[cfg(feature = "f16")]
    fn f32_reference(w0: f32, g: f32, config: &AdaGradConfig, steps: usize) -> f32 {
        let lr = config.lr as f32;
        let eps = config.eps as f32;
        let wd = config.weight_decay as f32;

        let mut w = w0;
        let mut acc = config.initial_accumulator_value as f32;

        for _ in 0..steps {
            let grad_wd = if wd > 0.0 { g + wd * w } else { g };
            acc += grad_wd * grad_wd;
            w -= lr * grad_wd / (acc.sqrt() + eps);
        }
        w
    }

    /// Run `steps` AdaGrad steps on a single-element parameter with a constant
    /// gradient, returning the final parameter.
    #[cfg(feature = "f16")]
    fn run_scalar_steps(
        client: &numr::runtime::cpu::CpuClient,
        param: Tensor<CpuRuntime>,
        grad: Tensor<CpuRuntime>,
        config: AdaGradConfig,
        steps: usize,
    ) -> Tensor<CpuRuntime> {
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut opt = AdaGrad::<CpuRuntime>::new(config);
        for _ in 0..steps {
            let mut grads = GradStore::new();
            grads.insert(id, grad.clone());
            opt.step(client, &mut params, &grads).unwrap();
        }
        params
            .remove(&id)
            .expect("param was inserted under this id")
    }

    /// The decisive test: a BF16 parameter under a realistic fine-tuning
    /// learning rate must actually move, and must track an F32 reference run.
    ///
    /// Without F32 master weights the parameter is returned unchanged, bit for
    /// bit, because AdaGrad's normalized step is ~`lr`, and `w - lr` rounds
    /// straight back to `w` in BF16.
    #[cfg(feature = "f16")]
    #[test]
    fn test_adagrad_bf16_parameter_actually_moves() {
        let (client, device) = cpu_setup();

        let w0 = 0.02f32;
        let g = 1.0f32;
        let steps = 128;
        let config = AdaGradConfig {
            lr: 2e-5,
            ..Default::default()
        };

        // Premise: AdaGrad's first step is `lr * g / sqrt(g^2)` = `lr`, which is
        // below BF16's resolution at this weight, so a BF16-only update rounds
        // it away. Later steps are smaller still.
        assert_eq!(
            half::bf16::from_f32(w0 - config.lr as f32).to_bits(),
            half::bf16::from_f32(w0).to_bits(),
            "test premise broken: a single step is representable in BF16"
        );

        let param =
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(w0)], &[1], &device)
                .unwrap();
        let grad = Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(g)], &[1], &device)
            .unwrap();
        let out = run_scalar_steps(&client, param, grad, config.clone(), steps);

        assert_eq!(
            out.dtype(),
            DType::BF16,
            "the model's parameter must stay BF16 — only the update is F32"
        );

        // Measure from the BF16-ROUNDED start: `bf16(0.02)` is 0.0200195, and
        // the 1.95e-5 gap to the literal is the same order as the step.
        let started = half::bf16::from_f32(w0).to_f32();
        let got = out.to_vec::<half::bf16>()[0].to_f32();
        assert!(
            started - got > 1e-4,
            "BF16 parameter did not move: started {started}, ended {got} after {steps} steps"
        );

        let expected = f32_reference(started, half::bf16::from_f32(g).to_f32(), &config, steps);
        assert!(
            (got - expected).abs() < 1e-4,
            "BF16 run must track the F32 reference: got {got} expected {expected}"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_adagrad_bf16_state_and_master_are_f32() {
        let (client, device) = cpu_setup();

        let param =
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(0.02)], &[1], &device)
                .unwrap();
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(1.0)], &[1], &device)
                .unwrap(),
        );

        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig {
            lr: 2e-5,
            ..Default::default()
        });
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        let master = state
            .master
            .as_ref()
            .expect("a BF16 param must get an F32 master copy");
        assert_eq!(master.dtype(), DType::F32);
        assert_eq!(
            state.acc.dtype(),
            DType::F32,
            "the accumulator must be F32 for a BF16 param — a narrow sum of squares saturates"
        );

        // The master already carries the step the BF16 parameter cannot show.
        let started = half::bf16::from_f32(0.02).to_f32();
        let moved = started - master.to_vec::<f32>()[0];
        assert!(
            moved > 1e-5,
            "master weight did not take the step: moved by {moved}"
        );
        assert_eq!(
            params.get(&id).unwrap().dtype(),
            DType::BF16,
            "the caller's parameter must stay BF16"
        );
    }

    #[test]
    fn test_adagrad_f32_allocates_no_master_copy() {
        let (client, device) = cpu_setup();

        let param = Tensor::<CpuRuntime>::try_from_slice(&[0.02f32], &[1], &device).unwrap();
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<CpuRuntime>::try_from_slice(&[0.001f32], &[1], &device).unwrap(),
        );

        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig {
            lr: 0.1,
            ..Default::default()
        });
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        assert!(
            state.master.is_none(),
            "an F32 parameter must not get a master copy"
        );
        assert_eq!(state.acc.dtype(), DType::F32);
    }
}
