//! Single pre-norm transformer layer for VoxCPM2's shared bidirectional
//! MiniCPM4 block stack, used by both `feat_encoder` (`local_encoder`) and
//! the local DiT (`feat_decoder`).
//!
//! Same pre-norm-attention-residual, pre-norm-MLP-residual sequence as
//! `LlamaBlock` (`RmsNorm` -> attn -> add, `RmsNorm` -> MLP -> add), with the
//! two differences this checkpoint requires: attention is
//! [`BidirectionalAttention`] (bidirectional GQA, not `LlamaAttention`'s
//! always-causal path) and residuals are plain adds — `use_mup` is `false`
//! on this checkpoint, so no muP `scale_depth/sqrt(num_layers)` factor is
//! applied (unlike some MiniCPM-lineage ports that assume it is).

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
use crate::model::traits::ModelClient;
use crate::nn::{Module, RmsNorm, RoPE, child_params, extend_named};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

pub struct BidirectionalLayer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: BidirectionalAttention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: BidirectionalMlp<R>,
}

impl<R: Runtime<DType = DType>> BidirectionalLayer<R> {
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` comes from the `MaybeLoraLinear` projections
        // inside the attention and MLP sub-blocks; the norms below need
        // nothing extra.
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }
}

/// Names ARE the field names (`input_layernorm`, `self_attn.*`,
/// `post_attention_layernorm`, `mlp.*`) — this matches the shared
/// `{layer_prefix}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::bidirectional::loader`]) exactly, so the
/// owning `LocalEncoder`/`LocalDit` need only prefix by `{layer_prefix}` (a
/// numeric layer index under `encoder.layers`/`decoder.layers`) to reach the
/// full checkpoint key.
impl<R: Runtime<DType = DType>> Module<R> for BidirectionalLayer<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.input_layernorm);
        params.extend(child_params(&self.self_attn));
        params.extend(child_params(&self.post_attention_layernorm));
        params.extend(child_params(&self.mlp));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "input_layernorm",
            self.input_layernorm.named_parameters(),
        );
        extend_named(&mut params, "self_attn", self.self_attn.named_parameters());
        extend_named(
            &mut params,
            "post_attention_layernorm",
            self.post_attention_layernorm.named_parameters(),
        );
        extend_named(&mut params, "mlp", self.mlp.named_parameters());
        params
    }
}
