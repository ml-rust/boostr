//! RMS Normalization module
//!
//! RMSNorm: output = x * rsqrt(mean(x^2, last_dim) + eps) * weight
//! Used in LLaMA, Mistral, and other modern architectures.
//! Delegates to numr's fused `var_rms_norm` for single-kernel forward + autograd.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_rms_norm};
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// RMS Normalization layer
///
/// weight: `[hidden_size]`
pub struct RmsNorm<R: Runtime> {
    weight: Var<R>,
    eps: f32,
}

impl<R: Runtime> RmsNorm<R> {
    /// Create a new RmsNorm layer
    pub fn new(weight: Tensor<R>, eps: f32, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            eps,
        }
    }

    /// Forward: x * rsqrt(mean(x^2) + eps) * weight
    ///
    /// Uses numr's fused NormalizationOps kernel (single kernel launch).
    /// input: `[..., hidden_size]`, output: same shape
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime,
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        var_rms_norm(input, &self.weight, self.eps, client).map_err(Error::Numr)
    }

    /// Fused residual add + RMS norm: rms_norm(x + residual, weight, eps)
    ///
    /// Returns `(normed, pre_norm)` where `pre_norm = x + residual`.
    /// Single kernel launch instead of separate add + norm.
    /// Uses the raw NormalizationOps directly (bypasses autograd — for inference).
    pub fn fused_add_forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        residual: &Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        R: Runtime,
        C: RuntimeClient<R> + NormalizationOps<R>,
    {
        let (normed, pre_norm) = client
            .fused_add_rms_norm(
                x.tensor(),
                residual.tensor(),
                self.weight.tensor(),
                self.eps,
            )
            .map_err(Error::Numr)?;
        Ok((Var::new(normed, false), Var::new(pre_norm, false)))
    }

    /// Get the weight parameter
    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_rmsnorm_output_shape() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let norm = RmsNorm::new(weight, 1e-5, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, 4], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_rmsnorm_values() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let norm = RmsNorm::new(weight, 1e-6, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();

        // RMS = sqrt(mean([1,4,9,16])) = sqrt(7.5) ≈ 2.7386
        // normed = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
        let rms = (7.5f32).sqrt();
        for (i, &val) in data.iter().enumerate() {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (val - expected).abs() < 1e-4,
                "idx={i}: got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_rmsnorm_with_scale() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[2.0f32; 4], &[4], &device);
        let norm = RmsNorm::new(weight, 1e-6, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();

        let rms = (7.5f32).sqrt();
        for (i, &val) in data.iter().enumerate() {
            let expected = 2.0 * (i as f32 + 1.0) / rms;
            assert!((val - expected).abs() < 1e-4);
        }
    }
}
