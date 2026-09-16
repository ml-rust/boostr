//! `AdamWConfig`, per-parameter state, and the `AdamW` struct's bookkeeping
//! methods. The `step` algorithm itself lives in `super::step`.

use numr::dtype::DType;
use numr::runtime::Runtime;
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
pub(super) struct ParamState<R: Runtime> {
    pub(super) m: Tensor<R>,
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
    pub(super) config: AdamWConfig,
    pub(super) state: HashMap<TensorId, ParamState<R>>,
    pub(super) timestep: u64,
}

impl<R: Runtime<DType = DType>> AdamW<R> {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
            timestep: 0,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
