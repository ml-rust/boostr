//! RMS Normalization module
//!
//! RMSNorm: output = x * rsqrt(mean(x^2, last_dim) + eps) * weight
//! Used in LLaMA, Mistral, and other modern architectures.
//! Composed from numr autograd primitives for training support.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add_scalar, var_mean, var_mul, var_sqrt};
use numr::dtype::DType;
use numr::ops::{ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// RMS Normalization layer
///
/// weight: `[hidden_size]`
pub struct RmsNorm<R: Runtime> {
    weight: Var<R>,
    eps: f64,
}

impl<R: Runtime> RmsNorm<R> {
    pub fn new(weight: Tensor<R>, eps: f32, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            eps: eps as f64,
        }
    }

    /// Forward: x * rsqrt(mean(x^2) + eps) * weight
    ///
    /// input: `[..., hidden_size]`, output: same shape
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + ReduceOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        let ndim = input.shape().len();
        let last_dim = ndim - 1;

        // x^2
        let x_sq = var_mul(input, input, client).map_err(Error::Numr)?;

        // mean(x^2, last_dim, keepdim=true)
        let mean_sq = var_mean(&x_sq, &[last_dim], true, client).map_err(Error::Numr)?;

        // mean(x^2) + eps
        let mean_sq_eps = var_add_scalar(&mean_sq, self.eps, client).map_err(Error::Numr)?;

        // sqrt(mean(x^2) + eps)
        let rms = var_sqrt(&mean_sq_eps, client).map_err(Error::Numr)?;

        // x / rms
        let normed = numr::autograd::var_div(input, &rms, client).map_err(Error::Numr)?;

        // normed * weight
        var_mul(&normed, &self.weight, client).map_err(Error::Numr)
    }

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
