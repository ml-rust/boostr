//! Block-quantized embedding — keeps `embed_tokens.weight` PACKED on device
//! and dequantizes only the rows a forward pass gathers.
//!
//! Split out of `embedding.rs` (which owns the dense [`Embedding`]) purely to
//! stay under this crate's `nn/*.rs` file-size limit; the two are one logical
//! unit and [`MaybeQuantEmbedding`] dispatches between them.

use crate::error::{Error, Result};
use crate::nn::embedding::Embedding;
use crate::nn::module::Module;
use crate::nn::weight::Weight;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Token embedding backed by a block-quantized weight table (inference-only —
/// mirrors [`crate::nn::linear::QuantLinear`]).
///
/// Keeps `embed_tokens.weight` PACKED on device (e.g. 123 MB for Q6_K
/// `[73448, 2048]` instead of 602 MB dense) and dequantizes only the rows a
/// forward pass actually gathers, via [`QuantTensor::gather_rows`].
pub struct QuantEmbedding<R: Runtime> {
    weight: QuantTensor<R>,
}

impl<R: Runtime> QuantEmbedding<R> {
    pub fn new(weight: QuantTensor<R>) -> Self {
        Self { weight }
    }

    /// Forward: gather rows from the packed weight, then dequantize only
    /// those rows.
    ///
    /// indices: `[...]` integer tensor, output: `[..., embed_dim]` — same
    /// contract as [`Embedding::forward`].
    ///
    /// `index_select` (which [`QuantTensor::gather_rows`] uses internally)
    /// requires a 1-D index tensor, while `indices` here may be N-D (e.g.
    /// `[batch, seq]`), so indices are flattened before gathering and the
    /// result is reshaped back to `indices.shape() ++ [embed_dim]`
    /// afterward.
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + IndexingOps<R> + DequantOps<R>,
        R::Client: IndexingOps<R>,
    {
        let index_shape = indices.shape().to_vec();
        let flat_len = indices.numel();
        let flat_indices = indices.contiguous()?.reshape(&[flat_len])?;

        let gathered_quant = self.weight.gather_rows(client, &flat_indices)?;
        let gathered = client.dequantize(&gathered_quant, DType::F32)?;

        let embed_dim = self.weight.shape()[1];
        let mut out_shape = index_shape;
        out_shape.push(embed_dim);
        Ok(gathered.reshape(&out_shape)?)
    }

    pub fn weight(&self) -> &QuantTensor<R> {
        &self.weight
    }
}

/// An embedding layer that works with either a standard or a block-quantized
/// weight table — mirrors [`crate::nn::linear::MaybeQuantLinear`] exactly.
///
/// During inference with GGUF models, `embed_tokens.weight` may ship
/// block-quantized (e.g. Q6_K) alongside the model's other quantized linear
/// weights. This enum lets model structs use a single field type regardless
/// of which the checkpoint provides.
pub enum MaybeQuantEmbedding<R: Runtime> {
    Standard(Embedding<R>),
    Quantized(QuantEmbedding<R>),
}

impl<R: Runtime> MaybeQuantEmbedding<R> {
    /// Construct from a `Weight` (standard or block-quantized) plus `trainable`.
    ///
    /// Unlike [`MaybeQuantLinear::from_weight`](crate::nn::linear::MaybeQuantLinear::from_weight),
    /// this returns `Result`: `Weight::DecomposedQuant` (AWQ/GPTQ) has no
    /// embedding meaning — there is no row-gather kernel for decomposed
    /// int4 storage — so it is a named error here rather than a silent third
    /// variant or a panic.
    pub fn from_weight(weight: Weight<R>, trainable: bool) -> Result<Self> {
        match weight {
            Weight::Standard(t) => Ok(Self::Standard(Embedding::new(t, trainable))),
            Weight::Quantized(qt) => Ok(Self::Quantized(QuantEmbedding::new(qt))),
            Weight::DecomposedQuant(_) => Err(Error::ModelError {
                reason: "decomposed quantized weights (AWQ/GPTQ) have no embedding \
                         row-gather kernel; only standard or block-quantized (GGUF) \
                         weights are supported for embeddings"
                    .into(),
            }),
        }
    }

    /// Forward pass: works for both standard and block-quantized weights.
    pub fn forward<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + IndexingOps<R> + DequantOps<R>,
        R::Client: IndexingOps<R>,
    {
        match self {
            Self::Standard(emb) => emb.forward(client, indices),
            Self::Quantized(qemb) => {
                let out = qemb.forward(client, indices)?;
                Ok(Var::new(out, false))
            }
        }
    }

    /// All trainable-capable parameters with their stable autograd IDs.
    ///
    /// The quantized variant is inference-only and therefore exposes no
    /// `Var` parameters — block-quantized storage has no gradient.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            Self::Standard(emb) => emb.parameters(),
            Self::Quantized(_) => Vec::new(),
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
            Self::Standard(emb) => emb.named_parameters(),
            Self::Quantized(_) => Vec::new(),
        }
    }
}

impl<R: Runtime> Module<R> for MaybeQuantEmbedding<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeQuantEmbedding::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeQuantEmbedding::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeQuantEmbedding::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeQuantEmbedding::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_maybe_quant_embedding_quantized_forward_matches_dequant_then_gather() {
        use crate::quant::QuantFormat;
        use crate::quant::traits::QuantizeOps;

        let (client, device) = cpu_setup();
        // vocab=4, dim=256 (one Q6_K block per row)
        let source: Vec<f32> = (0..4 * 256).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let table = Tensor::<CpuRuntime>::from_slice(&source, &[4, 256], &device).unwrap();

        let qt = client.quantize(&table, QuantFormat::Q6K).unwrap();
        let maybe_quant = MaybeQuantEmbedding::from_weight(Weight::Quantized(qt), false).unwrap();
        assert!(matches!(maybe_quant, MaybeQuantEmbedding::Quantized(_)));

        // Reference: dequantize the whole table first, then gather with the
        // ordinary dense `Embedding` — this is the same quantization error as
        // the gather-then-dequantize path, so the two must agree exactly,
        // unlike comparing against the pre-quantization floats.
        let qt_for_reference = match &maybe_quant {
            MaybeQuantEmbedding::Quantized(qemb) => qemb.weight(),
            MaybeQuantEmbedding::Standard(_) => unreachable!(),
        };
        let dequantized_table = client.dequantize(qt_for_reference, DType::F32).unwrap();
        let reference_emb = Embedding::new(dequantized_table, false);

        // [2, 3] batched indices, matching test_embedding_batched's shape contract.
        let indices =
            Tensor::<CpuRuntime>::from_slice(&[3i64, 0, 2, 1, 3, 0], &[2, 3], &device).unwrap();

        let reference_out = reference_emb.forward(&client, &indices).unwrap();
        let quant_out = maybe_quant.forward(&client, &indices).unwrap();

        assert_eq!(quant_out.shape(), &[2, 3, 256]);
        assert_eq!(quant_out.shape(), reference_out.shape());
        assert!(!quant_out.requires_grad());
        let got_bits: Vec<u32> = quant_out
            .tensor()
            .to_vec::<f32>()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        let expected_bits: Vec<u32> = reference_out
            .tensor()
            .to_vec::<f32>()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        assert_eq!(got_bits, expected_bits);
    }

    #[test]
    fn test_maybe_quant_embedding_quantized_has_no_trainable_parameters() {
        use crate::quant::QuantFormat;
        use crate::quant::traits::QuantizeOps;

        let (client, device) = cpu_setup();
        let source: Vec<f32> = (0..4 * 256).map(|i| (i as f32) * 0.01).collect();
        let table = Tensor::<CpuRuntime>::from_slice(&source, &[4, 256], &device).unwrap();
        let qt = client.quantize(&table, QuantFormat::Q6K).unwrap();

        let maybe_quant = MaybeQuantEmbedding::from_weight(Weight::Quantized(qt), false).unwrap();

        assert!(maybe_quant.parameters().is_empty());
        assert!(maybe_quant.trainable_parameters().is_empty());
        assert!(maybe_quant.named_parameters().is_empty());
        assert!(Module::parameters(&maybe_quant).is_empty());
    }

    #[test]
    fn test_maybe_quant_embedding_rejects_decomposed_quant() {
        use crate::quant::decomposed::{DecomposedQuantMethod, DecomposedQuantTensor};

        let (_client, device) = cpu_setup();
        let qweight = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4, 1], &device).unwrap();
        let scales = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device).unwrap();
        let qzeros = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[1, 4], &device).unwrap();
        let dq = DecomposedQuantTensor::new(
            qweight,
            scales,
            qzeros,
            None,
            DecomposedQuantMethod::Awq { group_size: 128 },
            vec![4, 4],
        );

        let result = MaybeQuantEmbedding::<CpuRuntime>::from_weight(
            Weight::DecomposedQuant(Box::new(dq)),
            false,
        );
        assert!(result.is_err());
    }
}
