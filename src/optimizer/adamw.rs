//! AdamW optimizer
//!
//! Implements decoupled weight decay regularization (Loshchilov & Hutter, 2019).
//! Uses numr tensor ops directly — works on any backend without GPU↔CPU transfers.

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

/// AdamW optimizer with decoupled weight decay
///
/// Maintains first moment (m) and second moment (v) estimates per parameter.
/// State is lazily initialized on first `step()` call for each parameter.
///
/// For a parameter narrower than F32 (BF16/F16/FP8) the optimizer also holds an
/// F32 master copy and keeps `m`/`v` at F32: AdamW's normalized update is
/// smaller than BF16's resolution at fine-tuning learning rates, so updating
/// the narrow parameter directly rounds every step away and the model never
/// trains. See [`crate::optimizer::precision`].
///
/// Optimizer state is not persisted by this type — a resumed run rebuilds the
/// master copies from the checkpointed parameters on its first step.
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
    /// Parameters narrower than F32 (BF16/F16/FP8) are updated through an F32
    /// master copy held in optimizer state; the caller's map receives a cast of
    /// the updated master. F32/F64 parameters take the direct path with no
    /// extra allocation.
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

        // Collect params that have gradients, lazily initializing state (and
        // the F32 master copy for narrow-dtype params), and split them:
        //
        // - `direct`: the parameter, its gradient, and its state already share
        //   one dtype. These take the multi-tensor launch exactly as before.
        // - `widened`: the update needs an F32 master, an F32 gradient, or
        //   both. These run one parameter at a time so each temporary F32
        //   gradient is freed before the next one is allocated — batching them
        //   would hold an F32 copy of EVERY gradient at once.
        let mut direct: Vec<TensorId> = Vec::new();
        let mut widened: Vec<TensorId> = Vec::new();

        for (&id, param) in params.iter() {
            let Some(grad) = grads.get(id) else {
                continue;
            };
            // Entry rather than contains_key + insert: the master copy needs a
            // fallible `cast`, so `or_insert_with` cannot build it.
            let state = match self.state.entry(id) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let state_dtype = optimizer_state_dtype(param.dtype());
                    let m = Tensor::<R>::try_zeros(param.shape(), state_dtype, param.device())?;
                    let v = Tensor::<R>::try_zeros(param.shape(), state_dtype, param.device())?;
                    let master = if state_dtype == param.dtype() {
                        None
                    } else {
                        Some(client.cast(param, state_dtype)?)
                    };
                    entry.insert(ParamState { m, v, master })
                }
            };
            if state.master.is_some() || grad.dtype() != state.m.dtype() {
                widened.push(id);
            } else {
                direct.push(id);
            }
        }

        if !direct.is_empty() {
            // Build groups for multi-tensor launch
            let groups: Vec<(&Tensor<R>, &Tensor<R>, &Tensor<R>, &Tensor<R>)> = direct
                .iter()
                .map(|id| {
                    let param = params
                        .get(id)
                        .expect("id came from params.keys() while building `direct`");
                    let grad = grads
                        .get(*id)
                        .expect("`direct` only holds ids that have a gradient");
                    let state = self
                        .state
                        .get(id)
                        .expect("state was lazily initialized for every id in `direct`");
                    (param, grad, &state.m, &state.v)
                })
                .collect();

            let results =
                client.fused_multi_tensor_adamw(&groups, lr, beta1, beta2, eps, wd, step_size)?;

            // Write back results
            for (id, (new_param, new_m, new_v)) in direct.iter().zip(results) {
                let state_mut = self
                    .state
                    .get_mut(id)
                    .expect("state was lazily initialized for every id in `direct`");
                state_mut.m = new_m;
                state_mut.v = new_v;
                params.insert(*id, new_param);
            }
        }

        for id in widened {
            let param_dtype = params
                .get(&id)
                .expect("id came from params.keys() while building `widened`")
                .dtype();
            let grad = grads
                .get(id)
                .expect("`widened` only holds ids that have a gradient");

            let state = self
                .state
                .get(&id)
                .expect("state was lazily initialized for every id in `widened`");
            let state_dtype = state.m.dtype();
            let arith_param = match state.master.as_ref() {
                Some(master) => master,
                None => params
                    .get(&id)
                    .expect("id came from params.keys() while building `widened`"),
            };

            // Freed at the end of this iteration, so only one F32 gradient copy
            // is resident at a time.
            let widened_grad = if grad.dtype() == state_dtype {
                None
            } else {
                Some(client.cast(grad, state_dtype)?)
            };
            let arith_grad = widened_grad.as_ref().unwrap_or(grad);

            let (new_param, new_m, new_v) = client.fused_adamw_step(
                arith_param,
                arith_grad,
                &state.m,
                &state.v,
                lr,
                beta1,
                beta2,
                eps,
                wd,
                step_size,
            )?;

            let state_mut = self
                .state
                .get_mut(&id)
                .expect("state was lazily initialized for every id in `widened`");
            state_mut.m = new_m;
            state_mut.v = new_v;

            let updated = match state_mut.master.as_mut() {
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

    pub fn timestep(&self) -> u64 {
        self.timestep
    }

    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }

    /// Number of parameter state entries currently held by the optimizer.
    pub fn state_len(&self) -> usize {
        self.state.len()
    }

    /// Returns true if optimizer state exists for `id`.
    pub fn has_state(&self, id: TensorId) -> bool {
        self.state.contains_key(&id)
    }

    /// Stable parameter IDs with initialized optimizer state.
    pub fn state_ids(&self) -> impl Iterator<Item = TensorId> + '_ {
        self.state.keys().copied()
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
            + TypeConversionOps<R>
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
        let w_tensor =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)
                .unwrap();
        let w = Var::new(w_tensor, true);
        let w_id = w.id();

        let x = Tensor::<CpuRuntime>::try_from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], &device)
            .unwrap();
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
        let target =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)
                .unwrap();

        let w_init =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device)
                .unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[5.0f32, 5.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        // Create a zero gradient — weight decay should still shrink params
        let mut grads = GradStore::new();
        let zero_grad = Tensor::<CpuRuntime>::try_zeros(&[2], DType::F32, &device).unwrap();
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

    /// Replay `fused_adamw_f32`'s exact arithmetic for a single scalar.
    ///
    /// Used to pin the F32 path bit-for-bit: if the optimizer ever widens or
    /// rounds an F32 parameter, these bits change.
    fn f32_reference(w0: f32, g: f32, config: &AdamWConfig, steps: i32) -> f32 {
        let b1 = config.beta1 as f32;
        let b2 = config.beta2 as f32;
        let e = config.eps as f32;
        let decay = (config.lr * config.weight_decay) as f32;

        let mut w = w0;
        let mut m = 0.0f32;
        let mut v = 0.0f32;

        for t in 1..=steps {
            let bc1 = 1.0 - config.beta1.powi(t);
            let bc2 = 1.0 - config.beta2.powi(t);
            let step_size = (config.lr * bc2.sqrt() / bc1) as f32;

            m = b1 * m + (1.0 - b1) * g;
            v = b2 * v + (1.0 - b2) * g * g;
            let update = step_size * m / (v.sqrt() + e);
            w = w * (1.0 - decay) - update;
        }
        w
    }

    /// Run `steps` AdamW steps on a single-element parameter with a constant
    /// gradient, returning the final parameter as f32.
    fn run_scalar_steps(
        client: &numr::runtime::cpu::CpuClient,
        param: Tensor<CpuRuntime>,
        grad: Tensor<CpuRuntime>,
        config: AdamWConfig,
        steps: usize,
    ) -> Tensor<CpuRuntime> {
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut opt = AdamW::<CpuRuntime>::new(config);
        for _ in 0..steps {
            let mut grads = GradStore::new();
            grads.insert(id, grad.clone());
            opt.step(client, &mut params, &grads).unwrap();
        }
        params
            .remove(&id)
            .expect("param was inserted under this id")
    }

    #[test]
    fn test_adamw_f32_path_is_bit_exact() {
        let (client, device) = cpu_setup();

        let w0 = 0.02f32;
        let g = 0.001f32;
        let steps = 4;
        let config = AdamWConfig {
            lr: 2e-5,
            weight_decay: 0.01,
            ..Default::default()
        };

        let param = Tensor::<CpuRuntime>::try_from_slice(&[w0], &[1], &device).unwrap();
        let grad = Tensor::<CpuRuntime>::try_from_slice(&[g], &[1], &device).unwrap();
        let out = run_scalar_steps(&client, param, grad, config.clone(), steps);

        let expected = f32_reference(w0, g, &config, steps as i32);
        assert_eq!(
            out.dtype(),
            DType::F32,
            "an F32 parameter must stay F32 in the caller's map"
        );
        assert_eq!(
            out.to_vec::<f32>()[0].to_bits(),
            expected.to_bits(),
            "F32 AdamW must be bit-identical to the plain f32 kernel arithmetic: \
             got {} expected {}",
            out.to_vec::<f32>()[0],
            expected
        );
    }

    #[test]
    fn test_adamw_f32_allocates_no_master_copy() {
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

        let mut opt = AdamW::<CpuRuntime>::new(AdamWConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        assert!(
            state.master.is_none(),
            "an F32 parameter must not get a master copy"
        );
        assert_eq!(state.m.dtype(), DType::F32);
        assert_eq!(state.v.dtype(), DType::F32);
    }

    /// The decisive test: a BF16 parameter under a realistic fine-tuning
    /// learning rate must actually move, and must track an F32 reference run.
    ///
    /// Without F32 master weights the parameter is returned unchanged, bit for
    /// bit, because `w + delta_w` rounds straight back to `w` in BF16.
    #[cfg(feature = "f16")]
    #[test]
    fn test_adamw_bf16_parameter_actually_moves() {
        let (client, device) = cpu_setup();

        let w0 = 0.02f32;
        let g = 0.001f32;
        let steps = 32;
        let config = AdamWConfig {
            lr: 2e-5,
            weight_decay: 0.0,
            ..Default::default()
        };

        // Premise: one lr-sized step is below BF16's resolution at this weight.
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

        let got = out.to_vec::<half::bf16>()[0].to_f32();
        assert!(
            w0 - got > 1e-4,
            "BF16 parameter did not move: started {w0}, ended {got} after {steps} steps"
        );

        let expected = f32_reference(w0, g, &config, steps as i32);
        assert!(
            (got - expected).abs() < 1e-4,
            "BF16 run must track the F32 reference: got {got} expected {expected}"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_adamw_f16_parameter_actually_moves() {
        let (client, device) = cpu_setup();

        let w0 = 0.02f32;
        let g = 0.001f32;
        let steps = 32;
        let config = AdamWConfig {
            lr: 2e-5,
            weight_decay: 0.0,
            ..Default::default()
        };

        let param = Tensor::<CpuRuntime>::try_from_slice(&[half::f16::from_f32(w0)], &[1], &device)
            .unwrap();
        let grad =
            Tensor::<CpuRuntime>::try_from_slice(&[half::f16::from_f32(g)], &[1], &device).unwrap();
        let out = run_scalar_steps(&client, param, grad, config.clone(), steps);

        assert_eq!(out.dtype(), DType::F16);

        let got = out.to_vec::<half::f16>()[0].to_f32();
        let expected = f32_reference(w0, g, &config, steps as i32);
        assert!(
            w0 - got > 1e-5,
            "F16 parameter did not move: started {w0}, ended {got}"
        );
        assert!(
            (got - expected).abs() < 2e-5,
            "F16 run must track the F32 reference: got {got} expected {expected}"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_adamw_bf16_state_and_master_are_f32() {
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
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(0.001)], &[1], &device)
                .unwrap(),
        );

        let mut opt = AdamW::<CpuRuntime>::new(AdamWConfig {
            lr: 2e-5,
            weight_decay: 0.0,
            ..Default::default()
        });
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        assert_eq!(
            state.m.dtype(),
            DType::F32,
            "m must be F32 for a BF16 param"
        );
        assert_eq!(
            state.v.dtype(),
            DType::F32,
            "v must be F32 for a BF16 param"
        );
        let master = state
            .master
            .as_ref()
            .expect("a BF16 param must get an F32 master copy");
        assert_eq!(master.dtype(), DType::F32);

        // The master already carries the step the BF16 parameter cannot show.
        //
        // Measure from the BF16-ROUNDED start, not from the `0.02` literal:
        // `bf16(0.02)` is 0.0200195, so the literal sits 1.95e-5 away from
        // where the master actually began — the same order as the 2e-5 step
        // being measured, which would swamp it.
        let started = half::bf16::from_f32(0.02).to_f32();
        let moved = started - master.to_vec::<f32>()[0];
        assert!(
            moved > 1e-5,
            "master weight did not take the step: moved by {moved}"
        );
    }

    #[test]
    fn test_adamw_skips_missing_grads() {
        let (client, device) = cpu_setup();

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
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
