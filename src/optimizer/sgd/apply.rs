//! The per-parameter SGD update loop.

use super::types::{ParamState, Sgd};
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
use std::collections::hash_map::Entry;

impl<R: Runtime<DType = DType>> Sgd<R> {
    /// Apply one SGD update to every id in `param_ids` that has a gradient.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_updates<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &GradStore<R>,
        param_ids: Vec<TensorId>,
        lr: f64,
        momentum: f64,
        wd: f64,
        dampening: f64,
        nesterov: bool,
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
                    let master = init_master(client, param, state_dtype)?;
                    entry.insert(ParamState { buf: None, master })
                }
            };

            let arith_param = state.master.as_ref().unwrap_or(param);

            let widened_grad = widen_grad(client, grad, state_dtype)?;
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

            let updated = write_back(client, state.master.as_mut(), new_param, param_dtype)?;
            params.insert(id, updated);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::SgdConfig;
    use super::*;
    use crate::optimizer::traits::Optimizer;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_sgd_f32_allocates_no_master_copy() {
        let (client, device) = cpu_setup();

        let param = Tensor::<CpuRuntime>::from_slice(&[0.02f32], &[1], &device).unwrap();
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<CpuRuntime>::from_slice(&[0.001f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&[half::bf16::from_f32(w0)], &[1], &device).unwrap();
        let grad =
            Tensor::<CpuRuntime>::from_slice(&[half::bf16::from_f32(g)], &[1], &device).unwrap();
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
            Tensor::<CpuRuntime>::from_slice(&[half::bf16::from_f32(0.02)], &[1], &device).unwrap();
        let id = param.id();
        let mut params = HashMap::new();
        params.insert(id, param);

        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<CpuRuntime>::from_slice(&[half::bf16::from_f32(1.0)], &[1], &device).unwrap(),
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
}
