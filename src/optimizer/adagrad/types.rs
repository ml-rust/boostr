//! `AdaGradConfig`, per-parameter state, and the `AdaGrad` struct's basic
//! accessors. The `step` algorithm lives in `super::step`.

use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

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
pub(super) struct ParamState<R: Runtime> {
    /// Sum of squared gradients. Always held at the optimizer state dtype: the
    /// accumulator grows monotonically, so a narrow one saturates and the
    /// effective step size collapses.
    pub(super) acc: Tensor<R>,
    /// F32 master copy of the parameter, held ONLY when the parameter's own
    /// dtype is narrower than F32 (BF16/F16/FP8).
    ///
    /// The update runs against the master and a cast of the master is written
    /// back into the caller's `params` map, so the model keeps computing in its
    /// own dtype while the update arithmetic stays F32. For an F32 or F64
    /// parameter this is `None`: no copy, no extra allocation, and the numbers
    /// are bit-identical to a build without master weights.
    pub(super) master: Option<Tensor<R>>,
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
    pub(super) config: AdaGradConfig,
    pub(super) state: HashMap<TensorId, ParamState<R>>,
}

impl<R: Runtime<DType = numr::dtype::DType>> AdaGrad<R> {
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_adagrad_reset() {
        use crate::optimizer::traits::Optimizer;
        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig::default());
        opt.reset();
        assert!(opt.state.is_empty());
    }

    #[test]
    fn test_adagrad_set_lr() {
        use crate::optimizer::traits::Optimizer;
        let mut opt = AdaGrad::<CpuRuntime>::new(AdaGradConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }
}
