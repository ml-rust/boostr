//! `SgdConfig`, per-parameter state, and the `Sgd` struct's basic accessors.
//! The `step` algorithm lives in `super::step`.

use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

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
pub(super) struct ParamState<R: Runtime> {
    /// Momentum (velocity) buffer, created by the first step that runs with
    /// `momentum > 0`. Always held at the optimizer state dtype.
    pub(super) buf: Option<Tensor<R>>,
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
    pub(super) config: SgdConfig,
    pub(super) state: HashMap<TensorId, ParamState<R>>,
}

impl<R: Runtime<DType = numr::dtype::DType>> Sgd<R> {
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_sgd_reset() {
        use crate::optimizer::traits::Optimizer;
        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig {
            momentum: 0.9,
            ..Default::default()
        });
        opt.reset();
        assert!(opt.state.is_empty());
    }

    #[test]
    fn test_sgd_set_lr() {
        use crate::optimizer::traits::Optimizer;
        let mut opt = Sgd::<CpuRuntime>::new(SgdConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }
}
