//! Decomposed quantized linear layer for AWQ/GPTQ inference

use crate::error::Result;
use crate::quant::traits::QuantMatmulOps;
use numr::dtype::DType;
use numr::ops::{BinaryOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::method::DecomposedQuantMethod;
use super::tensor::DecomposedQuantTensor;

/// Linear layer backed by decomposed quantized weights (AWQ/GPTQ).
///
/// Inference-only: calls `int4_gemm` or `int4_gemm_gptq` depending on method.
/// Automatically casts input to F32 if needed (int4_gemm requires F32 activations).
pub struct DecomposedQuantLinear<R: Runtime> {
    weight: DecomposedQuantTensor<R>,
    bias: Option<Tensor<R>>,
}

impl<R: Runtime> DecomposedQuantLinear<R> {
    pub fn new(weight: DecomposedQuantTensor<R>, bias: Option<Tensor<R>>) -> Self {
        Self { weight, bias }
    }

    /// Forward: int4_gemm(input, qweight, scales, zeros, group_size) + bias
    ///
    /// If the input is not F32 (e.g. BF16 from embeddings/norms), it is cast
    /// to F32 before the GEMM since int4_gemm requires F32 activations.
    /// The output is cast back to the input's original dtype to maintain
    /// dtype consistency through the model (e.g. BF16 KV cache).
    pub fn forward<C>(&self, client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: QuantMatmulOps<R> + BinaryOps<R> + RuntimeClient<R> + TypeConversionOps<R>,
        R: Runtime<DType = DType>,
    {
        let input_dtype = input.dtype();

        // int4_gemm requires F32 input — cast if needed
        let input_f32 = if input_dtype != DType::F32 {
            client
                .cast(input, DType::F32)
                .map_err(crate::error::Error::Numr)?
        } else {
            input.clone()
        };

        let output = match self.weight.method {
            DecomposedQuantMethod::Awq { group_size } => client.int4_gemm(
                &input_f32,
                &self.weight.qweight,
                &self.weight.scales,
                &self.weight.qzeros,
                group_size,
            )?,
            DecomposedQuantMethod::Gptq { .. } => {
                return Err(crate::error::Error::ModelError {
                    reason: "GPTQ forward not yet implemented (needs g_idx tensor)".into(),
                });
            }
        };

        let output = match &self.bias {
            Some(bias) => client
                .add(&output, bias)
                .map_err(crate::error::Error::Numr)?,
            None => output,
        };

        // Cast back to input dtype to maintain consistency (e.g. BF16 activations)
        if input_dtype != DType::F32 {
            client
                .cast(&output, input_dtype)
                .map_err(crate::error::Error::Numr)
        } else {
            Ok(output)
        }
    }

    pub fn weight(&self) -> &DecomposedQuantTensor<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<R>> {
        self.bias.as_ref()
    }
}
