//! Linear and quantized linear layers

use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::weight::Weight;
use crate::quant::decomposed::DecomposedQuantLinear;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_add, var_matmul, var_reshape, var_transpose};
use numr::dtype::DType;
use numr::ops::{BinaryOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

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

    /// Create from tensors while preserving stable autograd IDs.
    ///
    /// Use this when rebuilding a layer from optimizer-updated tensors so the
    /// optimizer state keyed by `TensorId` remains attached to the same logical
    /// parameters across steps.
    pub fn with_ids(
        weight: Tensor<R>,
        weight_id: TensorId,
        bias: Option<(Tensor<R>, TensorId)>,
        trainable: bool,
    ) -> Self {
        Self {
            weight: Var::with_id(weight, weight_id, trainable),
            bias: bias.map(|(b, id)| Var::with_id(b, id, trainable)),
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
        let input_shape = input.shape().to_vec();

        if input_shape.len() <= 2 {
            let output = var_matmul(input, &w_t, client).map_err(crate::error::Error::Numr)?;
            return match &self.bias {
                Some(bias) => var_add(&output, bias, client).map_err(crate::error::Error::Numr),
                None => Ok(output),
            };
        }

        let last_axis = input_shape.len() - 1;
        let in_features = input_shape[last_axis];
        let leading: usize = input_shape[..last_axis].iter().product();
        let flat_input =
            var_reshape(input, &[leading, in_features]).map_err(crate::error::Error::Numr)?;
        let flat_output =
            var_matmul(&flat_input, &w_t, client).map_err(crate::error::Error::Numr)?;
        let flat_output = match &self.bias {
            Some(bias) => var_add(&flat_output, bias, client).map_err(crate::error::Error::Numr)?,
            None => flat_output,
        };

        let weight_shape = self.weight.tensor().shape();
        if weight_shape.is_empty() {
            return Err(crate::error::Error::ModelError {
                reason: "linear weight must have at least one dimension".into(),
            });
        }
        let mut output_shape = input_shape;
        output_shape[last_axis] = weight_shape[0];
        var_reshape(&flat_output, &output_shape).map_err(crate::error::Error::Numr)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.bias.as_ref()
    }

    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = vec![(self.weight.id(), &self.weight)];
        if let Some(bias) = &self.bias {
            params.push((bias.id(), bias));
        }
        params
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }
}

impl<R: Runtime> Module<R> for Linear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = vec![self.weight()];
        if let Some(bias) = self.bias() {
            params.push(bias);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = vec![("weight".to_string(), self.weight())];
        if let Some(bias) = self.bias() {
            params.push(("bias".to_string(), bias));
        }
        params
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
    DecomposedQuant(Box<DecomposedQuantLinear<R>>),
}

impl<R: Runtime> MaybeQuantLinear<R> {
    /// Construct from a `Weight` (standard, quantized, or decomposed) plus optional bias tensor.
    pub fn from_weight(weight: Weight<R>, bias: Option<Tensor<R>>) -> Self {
        match weight {
            Weight::Standard(t) => Self::Standard(Linear::new(t, bias, false)),
            Weight::Quantized(qt) => Self::Quantized(QuantLinear::new(qt, bias)),
            Weight::DecomposedQuant(dq) => {
                Self::DecomposedQuant(Box::new(DecomposedQuantLinear::new(*dq, bias)))
            }
        }
    }

    /// The base weight, if it is `Var`-wrapped — i.e. only for the dense
    /// `Standard` variant. Block-quantized and decomposed storage carry no
    /// trainable `Var<R>`, so this is `None` for those, not a panic.
    pub fn weight(&self) -> Option<&Var<R>> {
        match self {
            Self::Standard(linear) => Some(linear.weight()),
            Self::Quantized(_) | Self::DecomposedQuant(_) => None,
        }
    }

    /// The base bias, if it is `Var`-wrapped. Mirrors [`Self::weight`]: only
    /// the dense `Standard` variant's bias is `Var`-wrapped; a quantized
    /// bias (when present) is a plain frozen `Tensor<R>`.
    pub fn bias(&self) -> Option<&Var<R>> {
        match self {
            Self::Standard(linear) => linear.bias(),
            Self::Quantized(_) | Self::DecomposedQuant(_) => None,
        }
    }

    /// Forward pass: works for standard, quantized, and decomposed quantized weights.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + DequantOps<R> + numr::ops::MatmulOps<R>,
    {
        match self {
            Self::Standard(linear) => linear.forward(client, input),
            // Forward always uses the fast quantized kernel. When `input`
            // needs no gradient (inference), the output stays a detached
            // leaf — zero extra allocation, unchanged from before. When it
            // does (QLoRA training), `attach_quant_linear_backward` wires up
            // a node whose backward dequantizes the FROZEN weight only then
            // — see `crate::quant::autograd` for why the base weight itself
            // never gets a gradient.
            Self::Quantized(qlinear) => {
                let out = qlinear.forward(client, input.tensor())?;
                if input.requires_grad() {
                    crate::quant::attach_quant_linear_backward(input, out, qlinear.weight())
                } else {
                    Ok(Var::new(out, false))
                }
            }
            // AWQ/GPTQ packed layouts have no elementwise dequant op in
            // `DequantOps` (only fused `int4_gemm`/`int4_gemm_gptq`/
            // `marlin_gemm`, which take an activation, not just the weight),
            // so there is no existing op to build a clean input-gradient
            // from without guessing at backward math. Left detached, same as
            // before, until such an op exists.
            Self::DecomposedQuant(dqlinear) => {
                let out = dqlinear.forward(client, input.tensor())?;
                Ok(Var::new(out, false))
            }
        }
    }

    /// Batched forward: compute multiple projections sharing the same input.
    ///
    /// When all layers are block-quantized, uses `quant_matmul_batch` to amortize
    /// activation preprocessing (e.g. Q8_1 quantization on CUDA).
    /// For decomposed quantized layers, falls back to individual forward passes.
    pub fn forward_batch<C>(
        layers: &[&MaybeQuantLinear<R>],
        client: &C,
        input: &Var<R>,
    ) -> Result<Vec<Var<R>>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + DequantOps<R> + numr::ops::MatmulOps<R>,
    {
        // Check if all are block-quantized (no bias) — enables batch path
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
            // Same detach-vs-attach split as the single-layer path: only
            // pay for the graph node when `input` actually needs a gradient.
            if input.requires_grad() {
                outputs
                    .into_iter()
                    .zip(weights)
                    .map(|(out, weight)| {
                        crate::quant::attach_quant_linear_backward(input, out, weight)
                    })
                    .collect()
            } else {
                Ok(outputs.into_iter().map(|t| Var::new(t, false)).collect())
            }
        } else {
            // Fallback: individual forward passes
            layers.iter().map(|l| l.forward(client, input)).collect()
        }
    }

    /// All trainable-capable parameters with their stable autograd IDs.
    ///
    /// Quantized variants are inference-only and therefore expose no `Var`
    /// parameters.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            Self::Standard(linear) => linear.parameters(),
            Self::Quantized(_) | Self::DecomposedQuant(_) => Vec::new(),
        }
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named standard parameters for checkpoint traversal.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        match self {
            Self::Standard(linear) => linear.named_parameters(),
            Self::Quantized(_) | Self::DecomposedQuant(_) => Vec::new(),
        }
    }
}

// `QuantTensor::shape` is only available under `DType = DType`, so `shape`
// lives in its own block rather than constraining every other method on
// `MaybeQuantLinear` (`from_weight`, `forward`, `parameters`) that does not
// need it.
impl<R: Runtime<DType = numr::dtype::DType>> MaybeQuantLinear<R> {
    /// Logical weight shape `[out_features, in_features]`, for every variant.
    ///
    /// A block-quantized or decomposed base has no `Var<R>` weight to read
    /// `.shape()` off of, but its logical element shape is tracked
    /// regardless — this is what lets a LoRA adapter size its low-rank
    /// factors from a frozen QUANTIZED base, without caring whether that
    /// base is dense or quantized.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Standard(linear) => linear.weight().tensor().shape(),
            Self::Quantized(qlinear) => qlinear.weight().shape(),
            Self::DecomposedQuant(dqlinear) => dqlinear.weight().shape(),
        }
    }
}

impl<R: Runtime> From<Linear<R>> for MaybeQuantLinear<R> {
    fn from(linear: Linear<R>) -> Self {
        Self::Standard(linear)
    }
}

impl<R: Runtime> Module<R> for MaybeQuantLinear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeQuantLinear::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeQuantLinear::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeQuantLinear::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeQuantLinear::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests;
