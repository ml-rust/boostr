//! Layer Normalization module
//!
//! LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
//! Composed from numr autograd primitives for training support.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_add_scalar, var_mean, var_mul, var_sqrt, var_sub};
use numr::dtype::DType;
use numr::ops::{ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Layer Normalization
///
/// weight (gamma): `[hidden_size]`
/// bias (beta): `[hidden_size]`
pub struct LayerNorm<R: Runtime> {
    weight: Var<R>,
    bias: Var<R>,
    eps: f64,
}

impl<R: Runtime> LayerNorm<R> {
    pub fn new(weight: Tensor<R>, bias: Tensor<R>, eps: f32, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: Var::new(bias, trainable),
            eps: eps as f64,
        }
    }

    /// Forward: (x - mean) / sqrt(var + eps) * weight + bias
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

        // mean(x, last_dim, keepdim=true)
        let mean = var_mean(input, &[last_dim], true, client).map_err(Error::Numr)?;

        // x - mean
        let centered = var_sub(input, &mean, client).map_err(Error::Numr)?;

        // variance = mean((x - mean)^2)
        let sq = var_mul(&centered, &centered, client).map_err(Error::Numr)?;
        let variance = var_mean(&sq, &[last_dim], true, client).map_err(Error::Numr)?;

        // sqrt(var + eps)
        let var_eps = var_add_scalar(&variance, self.eps, client).map_err(Error::Numr)?;
        let std = var_sqrt(&var_eps, client).map_err(Error::Numr)?;

        // (x - mean) / std
        let normed = numr::autograd::var_div(&centered, &std, client).map_err(Error::Numr)?;

        // normed * weight + bias
        let scaled = var_mul(&normed, &self.weight, client).map_err(Error::Numr)?;
        var_add(&scaled, &self.bias, client).map_err(Error::Numr)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

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
