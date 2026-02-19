//! Neural network module traits for parameter access and serialization.

use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

/// Core trait for neural network modules.
///
/// Provides access to trainable parameters for optimizers and
/// named parameters for checkpoint serialization.
///
/// Forward passes stay as inherent methods on each layer because
/// signatures differ (different client bounds, input types).
pub trait Module<R: Runtime> {
    /// All trainable parameters (for optimizer).
    fn parameters(&self) -> Vec<&Var<R>>;

    /// Named parameters (for checkpointing). Names use dot notation
    /// for nested modules: `"layers.0.attn.weight"`.
    fn named_parameters(&self) -> Vec<(String, &Var<R>)>;

    /// Total number of scalar parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|v| v.tensor().numel()).sum()
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

    // Demonstrate implementing Module for Linear
    impl Module<CpuRuntime> for Linear<CpuRuntime> {
        fn parameters(&self) -> Vec<&Var<CpuRuntime>> {
            let mut params = vec![self.weight()];
            if let Some(b) = self.bias() {
                params.push(b);
            }
            params
        }

        fn named_parameters(&self) -> Vec<(String, &Var<CpuRuntime>)> {
            let mut params = vec![("weight".to_string(), self.weight())];
            if let Some(b) = self.bias() {
                params.push(("bias".to_string(), b));
            }
            params
        }
    }

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
