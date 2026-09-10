use crate::error::Result;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::QuantMatmulOps;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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

    /// Cheap duplicate for a `'static` activation-checkpointing closure.
    ///
    /// This layer carries no `Var<R>` — a quantized weight is inference-only
    /// and never trains — so there is no `TensorId` to preserve.
    /// `QuantTensor::clone` is `Arc`-backed and cheap; `bias` is a frozen
    /// `Tensor<R>`, cloned like any other.
    pub fn alias(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
        }
    }
}
