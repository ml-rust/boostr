//! LoRA (Low-Rank Adaptation) layer.
//!
//! Adds a low-rank A*B decomposition to an existing linear layer:
//! output = base_linear(x) + (x @ A^T) @ B^T * scaling
//!
//! where A: [rank, in_features], B: [out_features, rank], scaling = alpha / rank.

use crate::error::Result;
use numr::autograd::{Var, var_matmul, var_transpose};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::Linear;

/// LoRA adapter wrapping a frozen base Linear layer.
pub struct LoraLinear<R: Runtime> {
    /// Frozen base linear layer
    base: Linear<R>,
    /// Low-rank down-projection: [rank, in_features]
    lora_a: Var<R>,
    /// Low-rank up-projection: [out_features, rank]
    lora_b: Var<R>,
    /// Scaling factor: alpha / rank
    scaling: f32,
}

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// Create a LoRA adapter around an existing linear layer.
    ///
    /// - `base`: The frozen base linear layer
    /// - `rank`: Low-rank dimension (typical: 4, 8, 16, 32)
    /// - `alpha`: Scaling factor (typical: rank or 2*rank)
    /// - `device`: Device to allocate LoRA weights on
    pub fn new(base: Linear<R>, rank: usize, alpha: f32, device: &R::Device) -> Self {
        let in_features = base.weight().tensor().shape()[1];
        let out_features = base.weight().tensor().shape()[0];

        // Initialize A with Kaiming uniform (simple LCG PRNG), B with zeros (standard LoRA init)
        let a_data = {
            let bound = (1.0 / in_features as f64).sqrt() as f32;
            let mut state: u64 = 42;
            let data: Vec<f32> = (0..rank * in_features)
                .map(|_| {
                    // Simple LCG for deterministic init
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
                    (u * 2.0 - 1.0) * bound
                })
                .collect();
            data
        };

        let lora_a = Var::new(
            Tensor::from_slice(&a_data, &[rank, in_features], device),
            true,
        );
        let lora_b = Var::new(
            Tensor::zeros(&[out_features, rank], DType::F32, device),
            true,
        );

        Self {
            base,
            lora_a,
            lora_b,
            scaling: alpha / rank as f32,
        }
    }

    /// Create from pre-loaded LoRA weights.
    pub fn from_weights(base: Linear<R>, lora_a: Tensor<R>, lora_b: Tensor<R>, alpha: f32) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            base,
            lora_a: Var::new(lora_a, false),
            lora_b: Var::new(lora_b, false),
            scaling: alpha / rank as f32,
        }
    }

    /// Forward: base(x) + (x @ A^T @ B^T) * scaling
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        let base_out = self.base.forward(client, input)?;

        // LoRA path: input @ A^T @ B^T * scaling
        let a_t = var_transpose(&self.lora_a).map_err(crate::error::Error::Numr)?;
        let lora_mid = var_matmul(input, &a_t, client).map_err(crate::error::Error::Numr)?;
        let b_t = var_transpose(&self.lora_b).map_err(crate::error::Error::Numr)?;
        let lora_out = var_matmul(&lora_mid, &b_t, client).map_err(crate::error::Error::Numr)?;

        // Scale and add
        let scaled = Var::new(
            lora_out
                .tensor()
                .mul_scalar(self.scaling as f64)
                .map_err(crate::error::Error::Numr)?,
            lora_out.requires_grad(),
        );

        let result = Var::new(
            base_out
                .tensor()
                .add(scaled.tensor())
                .map_err(crate::error::Error::Numr)?,
            base_out.requires_grad() || scaled.requires_grad(),
        );

        Ok(result)
    }

    /// Get reference to the base linear layer.
    pub fn base(&self) -> &Linear<R> {
        &self.base
    }

    /// Get LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.tensor().shape()[0]
    }

    /// Get scaling factor.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_lora_linear_creation() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight: Tensor<CpuRuntime> = Tensor::zeros(&[64, 32], DType::F32, &device);
        let base = Linear::new(weight, None, false);
        let lora = LoraLinear::new(base, 8, 16.0, &device);
        assert_eq!(lora.rank(), 8);
        assert!((lora.scaling() - 2.0).abs() < 1e-6); // alpha/rank = 16/8 = 2
    }
}
