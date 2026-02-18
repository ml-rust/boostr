//! Linear and quantized linear layers

use crate::error::Result;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_add, var_matmul, var_transpose};
use numr::ops::{BinaryOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Dense linear layer: output = input @ weight^T + bias
///
/// Uses `Var<R>` throughout — autograd works during training,
/// near-zero overhead during inference.
pub struct Linear<R: Runtime> {
    weight: Var<R>,
    bias: Option<Var<R>>,
}

impl<R: Runtime> Linear<R> {
    /// Create from loaded tensors. `trainable` controls gradient tracking.
    pub fn new(weight: Tensor<R>, bias: Option<Tensor<R>>, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: bias.map(|b| Var::new(b, trainable)),
        }
    }

    /// Forward: input @ weight^T + bias
    ///
    /// input: `[..., in_features]`, output: `[..., out_features]`
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let w_t = var_transpose(&self.weight).map_err(crate::error::Error::Numr)?;
        let output = var_matmul(input, &w_t, client).map_err(crate::error::Error::Numr)?;
        match &self.bias {
            Some(bias) => var_add(&output, bias, client).map_err(crate::error::Error::Numr),
            None => Ok(output),
        }
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.bias.as_ref()
    }
}

/// Quantized linear layer (inference-only — quantized weights don't train)
///
/// Uses `QuantTensor<R>` for weights and raw `Tensor<R>` for activations.
pub struct QuantLinear<R: Runtime> {
    weight: QuantTensor<R>,
    bias: Option<Tensor<R>>,
}

impl<R: Runtime> QuantLinear<R> {
    pub fn new(weight: QuantTensor<R>, bias: Option<Tensor<R>>) -> Self {
        Self { weight, bias }
    }

    /// Forward: quant_matmul(input, weight) + bias
    ///
    /// input: `[..., in_features]`, output: `[..., out_features]`
    pub fn forward<C>(&self, client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: QuantMatmulOps<R> + BinaryOps<R> + RuntimeClient<R>,
    {
        let output = client.quant_matmul(input, &self.weight)?;
        match &self.bias {
            Some(bias) => client.add(&output, bias).map_err(crate::error::Error::Numr),
            None => Ok(output),
        }
    }

    pub fn weight(&self) -> &QuantTensor<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<R>> {
        self.bias.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_linear_output_shape() {
        let (client, device) = cpu_setup();
        // weight: [out=4, in=3]
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device);
        let linear = Linear::new(weight, None, false);

        // input: [2, 3]
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device),
            false,
        );
        let out = linear.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn test_linear_with_bias() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2], &device);
        let linear = Linear::new(weight, Some(bias), false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            false,
        );
        let out = linear.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        // [1,2] @ [[1,0],[0,1]] + [10,20] = [1,2] + [10,20] = [11,22]
        assert_eq!(data, vec![11.0, 22.0]);
    }

    #[test]
    fn test_linear_batched() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
        let linear = Linear::new(weight, None, false);

        // input: [4, 5, 3] — batched
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 60], &[4, 5, 3], &device),
            false,
        );
        let out = linear.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[4, 5, 2]);
    }
}
