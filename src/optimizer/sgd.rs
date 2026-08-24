//! SGD optimizer with momentum
//!
//! Implements stochastic gradient descent with optional momentum and weight decay.
//! Follows PyTorch's SGD semantics with Nesterov momentum support.

use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::precision::optimizer_state_dtype;
use crate::optimizer::traits::Optimizer;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

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

/// Per-parameter optimizer state
struct ParamState<R: Runtime> {
    /// Momentum (velocity) buffer, created by the first step that runs with
    /// `momentum > 0`. Always held at the optimizer state dtype.
    buf: Option<Tensor<R>>,
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
///
/// For a parameter narrower than F32 (BF16/F16/FP8) the optimizer also holds an
/// F32 master copy and keeps the velocity buffer at F32: at fine-tuning
/// learning rates `lr * g` is smaller than BF16's resolution, so updating the
/// narrow parameter directly rounds every step away and the model never trains.
/// See [`crate::optimizer::precision`].
///
/// Optimizer state is not persisted by this type — a resumed run rebuilds the
/// master copies from the checkpointed parameters on its first step.
pub struct Sgd<R: Runtime> {
    config: SgdConfig,
    state: HashMap<TensorId, ParamState<R>>,
}

impl<R: Runtime<DType = DType>> Sgd<R> {
    pub fn new(config: SgdConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
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
            + TypeConversionOps<R>
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
            let param = match params.get(&id) {
                Some(p) => p,
                None => continue,
            };

            let param_dtype = param.dtype();
            let state_dtype = optimizer_state_dtype(param_dtype);

            // Entry rather than contains_key + insert: the master copy needs a
            // fallible `cast`, so `or_insert_with` cannot build it.
            let state = match self.state.entry(id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let master = if state_dtype == param_dtype {
                        None
                    } else {
                        Some(client.cast(param, state_dtype)?)
                    };
                    entry.insert(ParamState { buf: None, master })
                }
            };

            // The kernels mutate the buffer they are handed in place (CUDA and
            // WGPU write through a storage-sharing clone of `param`), so the
            // master — never the narrow parameter — is what goes in.
            let arith_param = state.master.as_ref().unwrap_or(param);

            // CPU dispatch keys on the parameter dtype alone and reads the
            // gradient at that width, so a narrow gradient against an F32
            // master must be widened first. Dropped at the end of the
            // iteration: only one widened gradient is resident at a time.
            let widened_grad = if grad.dtype() == state_dtype {
                None
            } else {
                Some(client.cast(grad, state_dtype)?)
            };
            let arith_grad = widened_grad.as_ref().unwrap_or(grad);

            let (new_param, new_buf) = client.fused_sgd_step(
                arith_param,
                arith_grad,
                state.buf.as_ref(),
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            )?;

            if momentum > 0.0 {
                // The buffer is an exponential moving average of gradients: a
                // narrow buffer would round the tail terms away, and the CPU
                // dispatch would read it at the parameter's width. Pin it to
                // the state dtype instead of trusting the kernel's inference.
                state.buf = Some(if new_buf.dtype() == state_dtype {
                    new_buf
                } else {
                    client.cast(&new_buf, state_dtype)?
                });
            }

            // A fresh cast, never the returned handle: on CUDA and WGPU that
            // handle aliases the master's storage.
            let updated = match state.master.as_mut() {
                Some(master) => {
                    *master = new_param;
                    client.cast(master, param_dtype)?
                }
                None => new_param,
            };
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

        let w_tensor =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)
                .unwrap();
        let w_id = w_tensor.id();

        // grad = [0.1, 0.2, 0.3, 0.4]
        let grad = Tensor::<CpuRuntime>::try_from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2], &device)
            .unwrap();
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

        let target =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)
                .unwrap();
        let w_init =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device)
                .unwrap();
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

        let target =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)
                .unwrap();
        let w_init =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device)
                .unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[5.0f32, 5.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let zero_grad = Tensor::<CpuRuntime>::try_zeros(&[2], DType::F32, &device).unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
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
        assert!(opt.state.is_empty());
    }

    #[test]
    fn test_sgd_set_lr() {
        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }

    /// Plain f32 SGD, mirroring the kernel arithmetic element-wise.
    #[cfg(feature = "f16")]
    fn f32_reference(w0: f32, g: f32, config: &SgdConfig, steps: usize) -> f32 {
        let lr = config.lr as f32;
        let mom = config.momentum as f32;
        let damp = config.dampening as f32;
        let wd = config.weight_decay as f32;

        let mut w = w0;
        let mut buf = 0.0f32;
        let mut has_buf = false;

        for _ in 0..steps {
            let grad_wd = if wd > 0.0 { g + wd * w } else { g };
            let b = if mom > 0.0 && has_buf {
                mom * buf + (1.0 - damp) * grad_wd
            } else {
                grad_wd
            };
            buf = b;
            if mom > 0.0 {
                has_buf = true;
            }
            let update = if config.nesterov && mom > 0.0 {
                grad_wd + mom * b
            } else if mom > 0.0 {
                b
            } else {
                grad_wd
            };
            w -= lr * update;
        }
        w
    }

    /// Run `steps` SGD steps on a single-element parameter with a constant
    /// gradient, returning the final parameter.
    #[cfg(feature = "f16")]
    fn run_scalar_steps(
        client: &numr::runtime::cpu::CpuClient,
        param: Tensor<CpuRuntime>,
        grad: Tensor<CpuRuntime>,
        config: SgdConfig,
        steps: usize,
    ) -> Tensor<CpuRuntime> {
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut opt = Sgd::<CpuRuntime>::new(config);
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
    /// bit, because `w - lr * g` rounds straight back to `w` in BF16.
    #[cfg(feature = "f16")]
    #[test]
    fn test_sgd_bf16_parameter_actually_moves() {
        let (client, device) = cpu_setup();

        let w0 = 0.02f32;
        let g = 1.0f32;
        let steps = 32;
        let config = SgdConfig {
            lr: 2e-5,
            ..Default::default()
        };

        // Premise: one step of `lr * g` is below BF16's resolution at this
        // weight, so a BF16-only update rounds it away every time.
        assert_eq!(
            half::bf16::from_f32(w0 - config.lr as f32 * g).to_bits(),
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
    fn test_sgd_bf16_state_and_master_are_f32() {
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

        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig {
            lr: 2e-5,
            momentum: 0.9,
            ..Default::default()
        });
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        let master = state
            .master
            .as_ref()
            .expect("a BF16 param must get an F32 master copy");
        assert_eq!(master.dtype(), DType::F32);
        let buf = state
            .buf
            .as_ref()
            .expect("momentum > 0 must create a velocity buffer");
        assert_eq!(
            buf.dtype(),
            DType::F32,
            "the velocity buffer must be F32 for a BF16 param"
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
    fn test_sgd_f32_allocates_no_master_copy() {
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

        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        });
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        assert!(
            state.master.is_none(),
            "an F32 parameter must not get a master copy"
        );
        assert_eq!(
            state
                .buf
                .as_ref()
                .expect("momentum > 0 must create a velocity buffer")
                .dtype(),
            DType::F32
        );
    }
}
