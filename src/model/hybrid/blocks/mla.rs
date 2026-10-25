//! The hybrid model's MLA block: pre-norm, absorbed MLA over a latent KV
//! cache, residual, pre-norm, feed-forward, residual.

use super::ffn::FeedForward;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::{AbsorbedMla, RmsNorm};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// MLA block: the oxidizr MLA layer layout, `input_layernorm`,
/// `self_attn.*`, `post_attention_layernorm`, then `mlp.*` or `moe.*`.
pub(in crate::model::hybrid) struct MlaBlock<R: Runtime> {
    pub(in crate::model::hybrid) input_layernorm: RmsNorm<R>,
    pub(in crate::model::hybrid) attn: AbsorbedMla<R>,
    pub(in crate::model::hybrid) post_attention_layernorm: RmsNorm<R>,
    pub(in crate::model::hybrid) ffn: FeedForward<R>,
}

impl<R: Runtime<DType = DType>> MlaBlock<R> {
    /// One KV-cached step at absolute `position`, which must equal the
    /// cache's length. See [`AbsorbedMla::forward_inference`].
    pub(in crate::model::hybrid) fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        self.forward_with_mixer(client, x, |normed| {
            self.attn
                .forward_inference(client, normed, kv_cache, position)
        })
    }

    /// The block around `mixer`: pre-norm, `mixer`, residual, pre-norm,
    /// feed-forward, residual.
    ///
    /// `mixer` maps the normed `[B, S, hidden]` input to the attention
    /// output of the same shape. A batched decode step passes a mixer that
    /// attends each row over its own cache.
    pub(in crate::model::hybrid) fn forward_with_mixer<C>(
        &self,
        client: &C,
        x: &Var<R>,
        mixer: impl FnOnce(&Var<R>) -> Result<Var<R>>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn = mixer(&normed)?;
        let h = var_add(x, &attn, client).map_err(Error::Numr)?;
        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let ffn = self.ffn.forward(client, &normed)?;
        var_add(&h, &ffn, client).map_err(Error::Numr)
    }
}
