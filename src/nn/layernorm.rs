//! Layer Normalization module
//!
//! LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
//! Delegates to numr's fused `var_layer_norm` for single-kernel forward + autograd.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_layer_norm};
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Layer Normalization
///
/// weight (gamma): `[hidden_size]`
/// bias (beta): `[hidden_size]`
pub struct LayerNorm<R: Runtime> {
    weight: Var<R>,
    bias: Var<R>,
    eps: f32,
}

impl<R: Runtime> LayerNorm<R> {
    /// Create a new LayerNorm layer
    pub fn new(weight: Tensor<R>, bias: Tensor<R>, eps: f32, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: Var::new(bias, trainable),
            eps,
        }
    }

    /// Forward: (x - mean) / sqrt(var + eps) * weight + bias
    ///
    /// Uses numr's fused NormalizationOps kernel (single kernel launch).
    /// input: `[..., hidden_size]`, output: same shape
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime,
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        var_layer_norm(input, &self.weight, &self.bias, self.eps, client).map_err(Error::Numr)
    }

    /// Get the weight (gamma) parameter
    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    /// Get the bias (beta) parameter
    pub fn bias(&self) -> &Var<R> {
        &self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_layernorm_output_shape() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
        let norm = LayerNorm::new(weight, bias, 1e-5, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, 4], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_layernorm_zero_mean_unit_var() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
        let norm = LayerNorm::new(weight, bias, 1e-6, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();

        // mean should be ~0
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");

        // var should be ~1
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn test_layernorm_affine() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[2.0f32; 4], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[5.0f32; 4], &[4], &device);
        let norm = LayerNorm::new(weight, bias, 1e-6, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();

        // mean of output should be 5 (bias), since normed has mean 0
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!((mean - 5.0).abs() < 0.1, "mean={mean}");
    }
}
