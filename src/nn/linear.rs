//! Linear and quantized linear layers

use crate::error::Result;
use crate::nn::weight::Weight;
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

/// A linear layer that works with either standard or quantized weights.
///
/// During inference with GGUF models, some weights are quantized (Q4_K_M etc.)
/// while others (norms, embeddings) remain in full precision. This enum lets
/// model structs use a single field type for both cases.
pub enum MaybeQuantLinear<R: Runtime> {
    Standard(Linear<R>),
    Quantized(QuantLinear<R>),
}

impl<R: Runtime> MaybeQuantLinear<R> {
    /// Construct from a `Weight` (standard or quantized) plus optional bias tensor.
    pub fn from_weight(weight: Weight<R>, bias: Option<Tensor<R>>) -> Self {
        match weight {
            Weight::Standard(t) => Self::Standard(Linear::new(t, bias, false)),
            Weight::Quantized(qt) => Self::Quantized(QuantLinear::new(qt, bias)),
        }
    }

    /// Forward pass: works for both standard and quantized weights.
    ///
    /// For standard weights: uses autograd-compatible matmul.
    /// For quantized weights: extracts tensor, does quant_matmul, wraps result.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + QuantMatmulOps<R> + BinaryOps<R>,
        R::Client: TensorOps<R>,
    {
        match self {
            Self::Standard(linear) => linear.forward(client, input),
            Self::Quantized(qlinear) => {
                let out = qlinear.forward(client, input.tensor())?;
                Ok(Var::new(out, false))
            }
        }
    }

    /// Batched forward: compute multiple projections sharing the same input.
    ///
    /// When all layers are quantized, uses `quant_matmul_batch` to amortize
    /// activation preprocessing (e.g. Q8_1 quantization on CUDA).
    pub fn forward_batch<C>(
        layers: &[&MaybeQuantLinear<R>],
        client: &C,
        input: &Var<R>,
    ) -> Result<Vec<Var<R>>>
    where
        C: RuntimeClient<R> + TensorOps<R> + QuantMatmulOps<R> + BinaryOps<R>,
        R::Client: TensorOps<R>,
    {
        // Check if all are quantized (no bias) — enables batch path
        let all_quantized_no_bias = layers
            .iter()
            .all(|l| matches!(l, MaybeQuantLinear::Quantized(ql) if ql.bias().is_none()));

        if all_quantized_no_bias {
            let weights: Vec<&QuantTensor<R>> = layers
                .iter()
                .map(|l| match l {
                    MaybeQuantLinear::Quantized(ql) => ql.weight(),
                    _ => unreachable!(),
                })
                .collect();

            let outputs = client.quant_matmul_batch(input.tensor(), &weights)?;
            Ok(outputs.into_iter().map(|t| Var::new(t, false)).collect())
        } else {
            // Fallback: individual forward passes
            layers.iter().map(|l| l.forward(client, input)).collect()
        }
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
