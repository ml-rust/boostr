//! The two per-parameter update paths `AdamW::step` dispatches into: a
//! multi-tensor fast path when parameter, gradient and state already share
//! one dtype, and a widened path (through an F32 master copy) otherwise.

use super::types::{AdamW, ParamState};
use crate::error::Result;
use crate::ops::FusedOptimizerOps;
use crate::optimizer::precision::optimizer_state_dtype;
use crate::optimizer::{init_master, widen_grad, write_back};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> AdamW<R> {
    /// Lazily initialize per-parameter state (and, for narrow dtypes, the F32
    /// master copy) for every id in `params` that has a gradient, splitting
    /// them into the multi-tensor-eligible `direct` set and the `widened` set
    /// that needs per-parameter handling.
    pub(super) fn collect_and_init_state<C>(
        &mut self,
        client: &C,
        params: &HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
    ) -> Result<(Vec<TensorId>, Vec<TensorId>)>
    where
        C: RuntimeClient<R> + TypeConversionOps<R>,
    {
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
                    let m = Tensor::<R>::zeros(param.shape(), state_dtype, param.device())?;
                    let v = Tensor::<R>::zeros(param.shape(), state_dtype, param.device())?;
                    let master = init_master(client, param, state_dtype)?;
                    entry.insert(ParamState { m, v, master })
                }
            };
            if state.master.is_some() || grad.dtype() != state.m.dtype() {
                widened.push(id);
            } else {
                direct.push(id);
            }
        }

        Ok((direct, widened))
    }

    /// Multi-tensor fast path: parameter, gradient and state already share one
    /// dtype for every id in `direct`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn apply_direct<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
        direct: &[TensorId],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<()>
    where
        C: RuntimeClient<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + FusedOptimizerOps<R>,
    {
        if direct.is_empty() {
            return Ok(());
        }

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

        Ok(())
    }

    /// Per-parameter path: at least one of the parameter, its gradient, or its
    /// state needs widening (an F32 master copy, an F32 gradient, or both).
    /// Runs one parameter at a time so each temporary F32 gradient is freed
    /// before the next one is allocated — batching them would hold an F32
    /// copy of EVERY gradient at once.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_widened<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
        widened: Vec<TensorId>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
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

            let widened_grad = widen_grad(client, grad, state_dtype)?;
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

            let updated = write_back(client, state_mut.master.as_mut(), new_param, param_dtype)?;
            params.insert(id, updated);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{f32_reference, run_scalar_steps};
    use super::super::types::AdamWConfig;
    use super::*;
    use crate::test_utils::cpu_setup;

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

        let param = Tensor::<crate::CpuRuntime>::from_slice(&[w0], &[1], &device).unwrap();
        let grad = Tensor::<crate::CpuRuntime>::from_slice(&[g], &[1], &device).unwrap();
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

        let param = Tensor::<crate::CpuRuntime>::from_slice(&[0.02f32], &[1], &device).unwrap();
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<crate::CpuRuntime>::from_slice(&[0.001f32], &[1], &device).unwrap(),
        );

        let mut opt = AdamW::<crate::CpuRuntime>::new(AdamWConfig::default());
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
            Tensor::<crate::CpuRuntime>::from_slice(&[half::bf16::from_f32(w0)], &[1], &device)
                .unwrap();
        let grad =
            Tensor::<crate::CpuRuntime>::from_slice(&[half::bf16::from_f32(g)], &[1], &device)
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

        let param =
            Tensor::<crate::CpuRuntime>::from_slice(&[half::f16::from_f32(w0)], &[1], &device)
                .unwrap();
        let grad =
            Tensor::<crate::CpuRuntime>::from_slice(&[half::f16::from_f32(g)], &[1], &device)
                .unwrap();
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
            Tensor::<crate::CpuRuntime>::from_slice(&[half::bf16::from_f32(0.02)], &[1], &device)
                .unwrap();
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<crate::CpuRuntime>::from_slice(&[half::bf16::from_f32(0.001)], &[1], &device)
                .unwrap(),
        );

        let mut opt = AdamW::<crate::CpuRuntime>::new(AdamWConfig {
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
}
