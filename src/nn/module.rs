//! Neural network module traits for parameter access and serialization.

use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// Core trait for neural network modules.
///
/// Provides access to trainable parameters for optimizers and
/// named parameters for checkpoint serialization.
///
/// Forward passes stay as inherent methods on each layer because
/// signatures differ (different client bounds, input types).
pub trait Module<R: Runtime> {
    /// All parameters owned by the module.
    fn parameters(&self) -> Vec<&Var<R>>;

    /// Named parameters (for checkpointing). Names use dot notation
    /// for nested modules: `"layers.0.attn.weight"`.
    fn named_parameters(&self) -> Vec<(String, &Var<R>)>;

    /// All parameters with their stable autograd IDs.
    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .map(|var| (var.id(), var))
            .collect()
    }

    /// Trainable parameters with their stable autograd IDs.
    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters_with_ids()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Clone trainable parameter tensors keyed by their stable autograd IDs.
    ///
    /// `Tensor::clone()` creates a fresh tensor storage ID, so optimizers must use
    /// the returned map key — not the cloned tensor's own ID — as the canonical
    /// parameter identity.
    fn trainable_parameter_tensors(&self) -> HashMap<TensorId, Tensor<R>> {
        self.trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().clone()))
            .collect()
    }

    /// Total number of scalar parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|v| v.tensor().numel()).sum()
    }
}

/// Training/eval mode switching for modules with mode-dependent behavior.
///
/// Modules like Dropout and BatchNorm behave differently during training
/// vs inference. This trait provides a unified interface for toggling mode.
///
/// Modules without mode-dependent behavior (Linear, LayerNorm, RMSNorm, etc.)
/// do NOT need to implement this trait.
///
/// # Recursive mode setting
///
/// Container modules (e.g., a full transformer) should propagate `set_training`
/// to all child modules that implement `TrainMode`:
///
/// ```ignore
/// impl TrainMode for MyModel {
///     fn set_training(&mut self, training: bool) {
///         self.dropout1.set_training(training);
///         self.dropout2.set_training(training);
///     }
///     fn is_training(&self) -> bool {
///         self.dropout1.is_training()
///     }
/// }
/// ```
pub trait TrainMode {
    /// Set training mode. When `true`, stochastic layers (dropout, batch norm)
    /// are active. When `false`, they behave deterministically.
    fn set_training(&mut self, training: bool);

    /// Returns `true` if the module is in training mode.
    fn is_training(&self) -> bool;

    /// Convenience: set to training mode.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Convenience: set to eval mode.
    fn eval(&mut self) {
        self.set_training(false);
    }
}

/// State dict serialization for model checkpointing.
///
/// Compatible with SafeTensors format via `boostr::format::safetensors`.
pub trait StateDict<R: Runtime>: Module<R> {
    /// Export model state as name→tensor map.
    fn state_dict(&self) -> HashMap<String, Tensor<R>> {
        self.named_parameters()
            .into_iter()
            .map(|(name, var)| (name, var.tensor().clone()))
            .collect()
    }

    /// Load state from a name→tensor map.
    ///
    /// Returns error if required keys are missing or shapes don't match.
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor<R>>) -> crate::error::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_module_parameters() {
        let device = CpuDevice::new();
        let weight = numr::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
        let bias = numr::tensor::Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[2], &device);
        let linear = Linear::new(weight, Some(bias), true);

        assert_eq!(linear.parameters().len(), 2);
        assert_eq!(linear.num_parameters(), 8); // 6 + 2
    }

    #[test]
    fn test_named_parameters() {
        let device = CpuDevice::new();
        let weight = numr::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
        let linear = Linear::new(weight, None, false);

        let named = linear.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }
}
