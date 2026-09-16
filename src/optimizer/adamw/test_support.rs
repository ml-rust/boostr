//! Shared f32-reference helpers for the AdamW precision tests
//! (`step.rs`'s own tests and the `f16`-gated tests in `tests_narrow.rs`).

use super::types::{AdamW, AdamWConfig};
use numr::autograd::GradStore;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use std::collections::HashMap;

/// Replay `fused_adamw_f32`'s exact arithmetic for a single scalar.
///
/// Used to pin the F32 path bit-for-bit: if the optimizer ever widens or
/// rounds an F32 parameter, these bits change.
pub(super) fn f32_reference(w0: f32, g: f32, config: &AdamWConfig, steps: i32) -> f32 {
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
pub(super) fn run_scalar_steps(
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
