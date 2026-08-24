//! LAMB optimizer (Layer-wise Adaptive Moments for Batch training)
//!
//! You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes", 2020.
//! Layer-wise adaptive scaling enables stable training at very large batch sizes (32K+).
//! Used by Google for BERT pre-training and applicable to frontier-scale LLM training.
//!
//! Also supports LARS mode (Layer-wise Adaptive Rate Scaling, You et al., 2017)
//! by setting `use_adam = false`, which uses SGD-style momentum instead of Adam moments.

use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::precision::optimizer_state_dtype;
use crate::optimizer::traits::Optimizer;
use crate::optimizer::{init_master, widen_grad, write_back};
use crate::readback::scalar_f32;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

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
    /// First moment. Always held at the optimizer state dtype.
    m: Tensor<R>,
    /// Second moment. Always held at the optimizer state dtype: it sums SQUARED
    /// gradients, which a narrow dtype flushes toward zero.
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

/// LAMB optimizer with layer-wise adaptive trust ratios
///
/// Computes Adam (or SGD momentum) updates per parameter, then scales each
/// layer's update by `||param|| / ||update||` (the "trust ratio"). This
/// normalization keeps gradient magnitudes consistent across layers,
/// enabling stable training at batch sizes of 32K+.
///
/// For a parameter narrower than F32 (BF16/F16/FP8) the optimizer also holds an
/// F32 master copy and keeps `m` and `v` at F32: LAMB's update is normalized, so
/// the step is `lr * trust_ratio` in magnitude, which at fine-tuning learning
/// rates is below BF16's resolution and rounds straight back to the original
/// weight. The trust ratio itself is computed over the master, so its two norms
/// are exact rather than rounded to the parameter's width. See
/// [`crate::optimizer::precision`].
///
/// Optimizer state is not persisted by this type — a resumed run rebuilds the
/// master copies from the checkpointed parameters on its first step.
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
///
/// The sum of squares carries the tensor's own dtype, so it is read back
/// through [`scalar_f32`]. Under BF16 or F16 a direct `item::<f32>` over-runs
/// the buffer, and both trust-ratio operands would be garbage.
fn tensor_l2_norm<R, C>(client: &C, t: &Tensor<R>) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + TypeConversionOps<R>,
{
    let sq = client.mul(t, t)?;
    let sum_sq = client.sum(&sq, &[], false)?;
    Ok((scalar_f32(client, &sum_sq)? as f64).sqrt())
}

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

            // Entry rather than contains_key + insert: the moments and the
            // master copy all need fallible calls, so `or_insert_with` cannot
            // build them.
            let state = match self.state.entry(id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let m = Tensor::<R>::try_zeros(param.shape(), state_dtype, param.device())?;
                    let v = Tensor::<R>::try_zeros(param.shape(), state_dtype, param.device())?;
                    let master = init_master(client, param, state_dtype)?;
                    entry.insert(LambState { m, v, master })
                }
            };

            let arith_param = state.master.as_ref().unwrap_or(param);

            let widened_grad = widen_grad(client, grad, state_dtype)?;
            let arith_grad = widened_grad.as_ref().unwrap_or(grad);

            // Fused kernel computes: update vector + updated m, v
            let (update, new_m, new_v) = client.fused_lamb_step(
                arith_param,
                arith_grad,
                &state.m,
                &state.v,
                beta1,
                beta2,
                eps,
                wd,
                bc1,
                bc2,
            )?;

            // Trust ratio requires global reductions (can't fuse into per-element kernel).
            // Taken over the master: `tensor_l2_norm` sums squares in the tensor's
            // own dtype, so a BF16 parameter would round the sum of squares and
            // perturb the ratio. Over the F32 master it is exact.
            let param_norm = tensor_l2_norm(client, arith_param)?;
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
            let new_param = client.sub(arith_param, &scaled_update)?;

            state.m = new_m;
            state.v = new_v;

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

        let target =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)
                .unwrap();
        let w_init =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device)
                .unwrap();
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
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)
                .unwrap();
        let w_init =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device)
                .unwrap();
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
        let w_tensor =
            Tensor::<CpuRuntime>::try_from_slice(&[100.0f32, 100.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let grad = Tensor::<CpuRuntime>::try_from_slice(&[0.001f32, 0.001], &[2], &device).unwrap();
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

        let w_tensor = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let w_id = w_tensor.id();

        let mut params = HashMap::new();
        params.insert(w_id, w_tensor);

        let grads = GradStore::new();
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.step(&client, &mut params, &grads).unwrap();

        let updated = params.get(&w_id).unwrap().to_vec::<f32>();
        assert_eq!(updated, vec![1.0, 2.0]);
    }

    /// The trust ratio is `||param|| / ||update||`. Both operands come from
    /// `tensor_l2_norm`, so a reinterpreted readback corrupts the ratio and the
    /// effective learning rate with it. [3, 4] has L2 norm 5.0 exactly in F32,
    /// F64, BF16 and F16, so the expected value is exact at every dtype and a
    /// byte-reinterpretation result cannot pass.
    #[test]
    fn test_tensor_l2_norm_f32_value_is_unchanged() {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
        let norm = tensor_l2_norm(&client, &t).unwrap();
        assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
    }

    /// F64 needs no feature flag, so this is the case that guards the fix under
    /// a plain `cargo test`. `item::<f32>` takes the low four bytes of the F64
    /// sum of squares, which for 25.0 are all zero, so the unfixed code reports
    /// a norm of 0.0 and LAMB falls back to a trust ratio of 1.0.
    #[test]
    fn test_tensor_l2_norm_reads_an_f64_tensor_at_its_own_dtype() {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
        let wide = client.cast(&t, DType::F64).unwrap();
        let norm = tensor_l2_norm(&client, &wide).unwrap();
        assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
    }

    /// The mixed-precision case: needs `--features f16`.
    #[cfg(feature = "f16")]
    #[test]
    fn test_tensor_l2_norm_reads_narrow_tensors_at_their_own_dtype() {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();

        for dtype in [DType::BF16, DType::F16] {
            let narrow = client.cast(&t, dtype).unwrap();
            assert_eq!(narrow.dtype(), dtype);
            let norm = tensor_l2_norm(&client, &narrow).unwrap();
            assert!(
                (norm - 5.0).abs() < 1e-2,
                "{dtype:?}: expected 5.0, got {norm}"
            );
        }
    }

    #[test]
    fn test_lamb_reset() {
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.reset();
        assert_eq!(opt.timestep(), 0);
        assert!(opt.state.is_empty());
    }

    #[test]
    fn test_lamb_set_lr() {
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }

    /// Plain f32 LAMB on a single element, mirroring the kernel arithmetic and
    /// the trust-ratio scaling this module applies around it.
    #[cfg(feature = "f16")]
    fn f32_reference(w0: f32, g: f32, config: &LambConfig, steps: usize) -> f32 {
        let b1 = config.beta1 as f32;
        let b2 = config.beta2 as f32;
        let e = config.eps as f32;
        let w = config.weight_decay as f32;

        let mut p = w0;
        let mut m = 0.0f32;
        let mut v = 0.0f32;

        for t in 1..=steps {
            let bc1 = 1.0 - config.beta1.powi(t as i32);
            let bc2 = if config.use_adam {
                1.0 - config.beta2.powi(t as i32)
            } else {
                1.0
            };

            m = b1 * m + (1.0 - b1) * g;
            v = b2 * v + (1.0 - b2) * g * g;

            let m_hat = m / bc1 as f32;
            let v_hat = v / bc2 as f32;
            let adam_update = m_hat / (v_hat.sqrt() + e);
            let update = if w > 0.0 {
                adam_update + w * p
            } else {
                adam_update
            };

            let param_norm = ((p * p) as f64).sqrt();
            let update_norm = ((update * update) as f64).sqrt();
            let trust = if param_norm > 0.0 && update_norm > 0.0 {
                let ratio = param_norm / update_norm;
                match config.max_trust_ratio {
                    Some(max) => ratio.min(max),
                    None => ratio,
                }
            } else {
                1.0
            };

            p -= (config.lr * trust) as f32 * update;
        }
        p
    }

    /// Run `steps` LAMB steps on a single-element parameter with a constant
    /// gradient, returning the optimizer and the final parameter map.
    #[cfg(feature = "f16")]
    fn run_scalar_steps(
        client: &numr::runtime::cpu::CpuClient,
        param: Tensor<CpuRuntime>,
        grad: Tensor<CpuRuntime>,
        config: LambConfig,
        steps: usize,
    ) -> (Lamb<CpuRuntime>, HashMap<TensorId, Tensor<CpuRuntime>>) {
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut opt = Lamb::<CpuRuntime>::new(config);
        for _ in 0..steps {
            let mut grads = GradStore::new();
            grads.insert(id, grad.clone());
            opt.step(client, &mut params, &grads).unwrap();
        }
        (opt, params)
    }

    /// The decisive test: a BF16 parameter under a realistic fine-tuning
    /// learning rate must actually move, and must land on the BF16 grid point
    /// nearest the F32 reference run.
    ///
    /// Reasoning for the premise: with a constant gradient the Adam update is
    /// bias-corrected to ~1.0, so `||update|| = 1` and the trust ratio is
    /// `||param||` itself — 0.02 here. One step is therefore `lr * 0.02` =
    /// 4e-7, while BF16's ulp at 0.02 is 2^-14 = 6.1e-5, over a hundred times
    /// larger. Without an F32 master every step rounds back to the starting
    /// weight, forever. Over 512 steps the master decays by ~2e-4, which is
    /// ~3.3 ulps and cannot round back.
    #[cfg(feature = "f16")]
    #[test]
    fn test_lamb_bf16_parameter_actually_moves() {
        let (client, device) = cpu_setup();

        let w0 = 0.02f32;
        let g = 1.0f32;
        let steps = 512;
        let config = LambConfig {
            lr: 2e-5,
            weight_decay: 0.0,
            ..Default::default()
        };

        let started = half::bf16::from_f32(w0).to_f32();

        // Premise: a single step is below BF16's resolution at this weight.
        let one_step = config.lr as f32 * started;
        assert_eq!(
            half::bf16::from_f32(started - one_step).to_bits(),
            half::bf16::from_f32(started).to_bits(),
            "test premise broken: a single step is representable in BF16"
        );

        let param =
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(w0)], &[1], &device)
                .unwrap();
        let grad = Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(g)], &[1], &device)
            .unwrap();
        let param_id = param.id();
        let (_opt, params) = run_scalar_steps(&client, param, grad, config.clone(), steps);

        let out = params.get(&param_id).unwrap();
        assert_eq!(
            out.dtype(),
            DType::BF16,
            "the model's parameter must stay BF16 — only the update is F32"
        );

        let got = out.to_vec::<half::bf16>()[0].to_f32();
        assert!(
            started - got > 1e-4,
            "BF16 parameter did not move: started {started}, ended {got} after {steps} steps"
        );

        // The master is exact F32, so the written-back parameter is precisely the
        // reference rounded to BF16. Assert that equality directly rather than
        // against a hand-derived ulp tolerance — the grid spacing changes with the
        // binade, and an off-by-one-binade tolerance is how this assertion first
        // failed while the value itself was correct.
        let expected = f32_reference(started, half::bf16::from_f32(g).to_f32(), &config, steps);
        let expected_bf16 = half::bf16::from_f32(expected).to_f32();
        assert_eq!(
            got, expected_bf16,
            "BF16 run must equal the F32 reference rounded to BF16: got {got}, \
             reference {expected} rounds to {expected_bf16}"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_lamb_bf16_state_and_master_are_f32() {
        let (client, device) = cpu_setup();

        let param =
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(0.02)], &[1], &device)
                .unwrap();
        let grad =
            Tensor::<CpuRuntime>::try_from_slice(&[half::bf16::from_f32(1.0)], &[1], &device)
                .unwrap();
        let param_id = param.id();

        let config = LambConfig {
            lr: 2e-5,
            weight_decay: 0.0,
            ..Default::default()
        };
        let (opt, params) = run_scalar_steps(&client, param, grad, config, 1);

        let state = opt
            .state
            .get(&param_id)
            .expect("state initialized on first step");
        let master = state
            .master
            .as_ref()
            .expect("a BF16 param must get an F32 master copy");
        assert_eq!(master.dtype(), DType::F32);
        assert_eq!(
            state.m.dtype(),
            DType::F32,
            "the first moment must be F32 for a BF16 param"
        );
        assert_eq!(
            state.v.dtype(),
            DType::F32,
            "the second moment must be F32 for a BF16 param — it sums squares"
        );

        // The master already carries the step the BF16 parameter cannot show.
        let started = half::bf16::from_f32(0.02).to_f32();
        let moved = started - master.to_vec::<f32>()[0];
        assert!(
            moved > 1e-7,
            "master weight did not take the step: moved by {moved}"
        );
        assert_eq!(
            params.get(&param_id).unwrap().dtype(),
            DType::BF16,
            "the caller's parameter must stay BF16"
        );
    }

    #[test]
    fn test_lamb_f32_allocates_no_master_copy() {
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

        let mut opt = Lamb::<CpuRuntime>::new(LambConfig {
            lr: 0.1,
            ..Default::default()
        });
        opt.step(&client, &mut params, &grads).unwrap();

        let state = opt.state.get(&id).expect("state initialized on first step");
        assert!(
            state.master.is_none(),
            "an F32 parameter must not get a master copy"
        );
        assert_eq!(state.m.dtype(), DType::F32);
        assert_eq!(state.v.dtype(), DType::F32);
    }
}
