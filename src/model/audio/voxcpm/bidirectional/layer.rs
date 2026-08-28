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
use crate::nn::{LoraTargets, Module, RmsNorm, RoPE, child_params, extend_named};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

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

    /// Delegate to `BidirectionalAttention::apply_lora` and
    /// `BidirectionalMlp::apply_lora`, summing their counts. `prefix` is the
    /// dotted path the owning `LocalEncoder`/`LocalDit` would pass to
    /// `extend_named` for this layer, extended here by `"self_attn"`/`"mlp"`
    /// exactly as `Module::named_parameters` extends it above. No
    /// zero-match check here: see `BidirectionalAttention::apply_lora`'s
    /// doc comment.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = self.self_attn.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "self_attn"),
        )?;
        adapted += self.mlp.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "mlp"),
        )?;
        Ok(adapted)
    }

    /// Delegate to `BidirectionalAttention::load_lora_parameters` and
    /// `BidirectionalMlp::load_lora_parameters`, summing their counts. No
    /// prefix needed — unlike [`Self::apply_lora`], lookup is by ID, not by
    /// dotted path.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = self.self_attn.load_lora_parameters(params)?;
        written += self.mlp.load_lora_parameters(params)?;
        Ok(written)
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
