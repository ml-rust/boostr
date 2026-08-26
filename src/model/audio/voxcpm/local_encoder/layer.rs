//! Single pre-norm transformer layer for `feat_encoder`.
//!
//! Same pre-norm-attention-residual, pre-norm-MLP-residual sequence as
//! `LlamaBlock` (`RmsNorm` -> attn -> add, `RmsNorm` -> MLP -> add), with the
//! two differences this checkpoint requires: attention is
//! [`LocalEncoderAttention`] (bidirectional GQA, not `LlamaAttention`'s
//! always-causal path) and residuals are plain adds — `use_mup` is `false`
//! on this checkpoint, so no muP `scale_depth/sqrt(num_layers)` factor is
//! applied (unlike some MiniCPM-lineage ports that assume it is).

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_encoder::attention::LocalEncoderAttention;
use crate::model::audio::voxcpm::local_encoder::mlp::LocalEncoderMlp;
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

pub struct LocalEncoderLayer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: LocalEncoderAttention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: LocalEncoderMlp<R>,
}

impl<R: Runtime<DType = DType>> LocalEncoderLayer<R> {
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
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
