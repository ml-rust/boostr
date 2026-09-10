//! [`DenseWeightSource`]: a [`WeightSource`] decorator that materializes every
//! packed weight to dense F32 on the way in.
//!
//! # Why the mode exists
//!
//! CONFORMANCE.md Section 7.1 requires the ACTIVATION CONTRACT to be matched
//! across two artifacts being compared. It is not matched by default. A TCF
//! declares an exact F32 contract, so its matmul runs f32 activations; a GGUF
//! declares no contract at all, so on CUDA at `m >= 2` the feature-major MMQ
//! path quantizes the activations to int8 before the tensor-core MMA. The
//! GGUF side then absorbs activation-quantization error the TCF side never
//! pays, and the gap is easy to misread as a weight-format difference.
//!
//! Loading through this decorator removes that confound: every packed weight
//! becomes a dense F32 tensor, both formats run the same dense F32 matmul,
//! and the only difference left between two artifacts is the weight VALUES —
//! the weight-encoding damage a format comparison is actually after.
//!
//! # What it costs, and what it is NOT for
//!
//! A dense stack costs what an unquantized checkpoint costs — the packed
//! path exists precisely to avoid that. This is a MEASUREMENT mode, never the
//! way to serve or fine-tune a quantized artifact.
//!
//! # No second dequantizer
//!
//! The conversion is [`DequantOps::dequantize`], the same op
//! `crate::quant::autograd`'s quantized-projection backward already runs on a
//! frozen weight, and the same kernels `quant_matmul` decodes with. Nothing
//! here reimplements a block or plane layout.

use super::source::WeightSource;
use crate::error::{Error, Result};
use crate::nn::Weight;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Any [`WeightSource`], presented as one that never yields a packed weight.
///
/// `load_named` is forwarded untouched — it is already the dense contract —
/// and `load_named_weight` is the whole of the change: a
/// [`Weight::Quantized`] is dequantized to F32 here, and a weight the source
/// already hands over dense passes straight through.
pub struct DenseWeightSource<'a, S, C> {
    inner: &'a mut S,
    client: &'a C,
}

impl<'a, S, C> DenseWeightSource<'a, S, C> {
    /// Wrap `inner`, dequantizing through `client`.
    pub fn new(inner: &'a mut S, client: &'a C) -> Self {
        Self { inner, client }
    }
}

impl<R, S, C> WeightSource<R> for DenseWeightSource<'_, S, C>
where
    R: Runtime<DType = DType>,
    S: WeightSource<R>,
    C: DequantOps<R>,
{
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.inner.load_named(name, device)
    }

    /// Dense, always. F32 is not a choice here: it is the dtype
    /// `quant_matmul` would have run the packed weight at, so a dense run
    /// that means to differ from the packed one in weight VALUES ONLY has to
    /// land on the same element type.
    fn load_named_weight(&mut self, name: &str, device: &R::Device) -> Result<Weight<R>> {
        match self.inner.load_named_weight(name, device)? {
            Weight::Quantized(packed) => Ok(Weight::Standard(
                self.client.dequantize(&packed, DType::F32)?,
            )),
            dense @ Weight::Standard(_) => Ok(dense),
            // `DequantOps` carries no elementwise dequantization for an
            // AWQ/GPTQ packed layout — only fused GEMMs that take an
            // activation — so there is no way to honour the contract for
            // one. No safetensors, GGUF or TCF source produces this
            // variant; if one ever does, it is named here rather than
            // silently left packed while the run reports itself dense.
            Weight::DecomposedQuant(_) => Err(Error::ModelError {
                reason: format!(
                    "{name}: dense-weight loading has no elementwise dequantization for a \
                     decomposed (AWQ/GPTQ) weight; only block- and plane-quantized weights \
                     can be materialized dense"
                ),
            }),
        }
    }
}
