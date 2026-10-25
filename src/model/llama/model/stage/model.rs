//! [`LlamaStage`]: the weights of one [`LayerSpan`] of a Llama, built from
//! a checkpoint that holds only the tensors the span reads.

use super::span::LayerSpan;
use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::llama::model::blocks::{BlockNorm, LlamaBlock};
use crate::model::llama::model::lm_head::build_lm_head;
use crate::model::llama::model::parts::{
    Dims, cast_rope, rope_of, take_embedding, take_final_norm, take_layer,
};
use crate::nn::{Embedding, MaybeQuantLinear, RoPE, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ReduceOps, ShapeOps};
use numr::runtime::Runtime;

/// The final norm and LM head of the span that ends at the last layer.
pub(super) struct StageHead<R: Runtime> {
    pub(super) norm: BlockNorm<R>,
    pub(super) lm_head: MaybeQuantLinear<R>,
}

/// One pipeline stage of a Llama: the layers of `span`, the embedding when
/// the span starts at layer 0, and the final norm and LM head when it ends
/// at the last layer. No other weight is loaded.
///
/// It runs the pipeline-stage forwards only: `forward_embed`,
/// `forward_layers_range` over held layers, and `forward_head`. Every
/// full-model path of [`LoadedModel`](crate::model::LoadedModel) refuses
/// it with the error [`Self::full_model_refusal`] returns.
pub struct LlamaStage<R: Runtime> {
    pub(super) config: ModelConfig,
    pub(super) span: LayerSpan,
    pub(super) embed_tokens: Option<Embedding<R>>,
    /// Layer `span.start() + i` at index `i`.
    pub(super) layers: Vec<LlamaBlock<R>>,
    pub(super) head: Option<StageHead<R>>,
    pub(super) rope: RoPE<R>,
}

impl<R: Runtime<DType = DType>> LlamaStage<R> {
    /// The stage of `span` from `vb`, which needs to hold only the tensors
    /// [`LayerSpan::reads`] names. Tensors of other layers are never taken.
    ///
    /// # Errors
    ///
    /// A config that does not validate, a span over another layer count,
    /// or a tensor the span reads that `vb` does not hold.
    pub fn from_varbuilder(
        vb: &mut VarBuilder<R>,
        config: &ModelConfig,
        span: LayerSpan,
    ) -> Result<Self>
    where
        R::Client: crate::quant::DequantOps<R>
            + numr::ops::TypeConversionOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
    {
        config.validate()?;
        if span.num_layers() != config.num_layers {
            return Err(Error::ModelError {
                reason: format!(
                    "a stage of {span} cannot load a model of {} layers",
                    config.num_layers
                ),
            });
        }
        let dims = Dims::of(config)?;
        let rope = rope_of(config, dims, vb.device())?;

        let mut model_vb = vb.pp("model");
        let reads_embedding =
            span.has_embedding() || (span.has_head() && config.tie_word_embeddings);
        let embedding = if reads_embedding {
            Some(take_embedding(&mut model_vb, config)?)
        } else {
            None
        };
        let layers = span
            .layers()
            .map(|i| take_layer(&mut model_vb, config, i, dims))
            .collect::<Result<Vec<_>>>()?;
        let norm = if span.has_head() {
            Some(take_final_norm(&mut model_vb, config)?)
        } else {
            None
        };

        let head = match norm {
            Some(norm) => {
                let embed = embedding.as_ref().map(|(e, name)| (e, name.as_str()));
                let lm_head = build_lm_head(vb, config, embed)?;
                Some(StageHead { norm, lm_head })
            }
            None => None,
        };
        // A head span of a tied model read the embedding for its head only.
        let embed_tokens = embedding.filter(|_| span.has_embedding()).map(|(e, _)| e);
        let rope = cast_rope(rope, &layers)?;
        Ok(Self {
            config: config.clone(),
            span,
            embed_tokens,
            layers,
            head,
            rope,
        })
    }

    /// The config of the whole model.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The layers this stage holds.
    pub fn span(&self) -> &LayerSpan {
        &self.span
    }

    /// The RoPE tables every held layer reads.
    pub fn rope(&self) -> &RoPE<R> {
        &self.rope
    }

    /// The dtype every forward computes in: that of the first held layer's
    /// input norm, an owned weight that is never quantized.
    pub fn dtype(&self) -> DType {
        self.layers
            .first()
            .map_or(DType::F32, |l| l.input_layernorm.weight().tensor().dtype())
    }

    /// The error a full-model path returns for this stage: `path` needs
    /// every layer, and this stage holds its span only.
    pub fn full_model_refusal(&self, path: &str) -> Error {
        Error::ModelError {
            reason: format!(
                "{path} runs the whole model, and this pipeline stage holds {} only. It runs \
                 forward_embed, forward_layers_range and forward_head. Load the model \
                 without a layer range to run {path}",
                self.span
            ),
        }
    }
}
