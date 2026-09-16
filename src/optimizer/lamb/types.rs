//! `LambConfig`, per-parameter state, and the `Lamb` struct's basic
//! accessors. The `step` algorithm lives in `super::step`.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

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

pub(super) struct LambState<R: Runtime> {
    /// First moment. Always held at the optimizer state dtype.
    pub(super) m: Tensor<R>,
    /// Second moment. Always held at the optimizer state dtype: it sums SQUARED
    /// gradients, which a narrow dtype flushes toward zero.
    pub(super) v: Tensor<R>,
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
    pub(super) config: LambConfig,
    pub(super) state: HashMap<TensorId, LambState<R>>,
    pub(super) timestep: u64,
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_lamb_reset() {
        use crate::optimizer::traits::Optimizer;
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.reset();
        assert_eq!(opt.timestep(), 0);
        assert!(opt.state.is_empty());
    }

    #[test]
    fn test_lamb_set_lr() {
        use crate::optimizer::traits::Optimizer;
        let mut opt = Lamb::<CpuRuntime>::new(LambConfig::default());
        opt.set_lr(0.05);
        assert_eq!(opt.lr(), 0.05);
    }
}
