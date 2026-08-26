//! Single pre-norm decoder layer for VoxCPM2's MiniCPM4.
//!
//! `RmsNorm` -> causal attention -> add, `RmsNorm` -> SwiGLU MLP -> add.
//!
//! Residuals are PLAIN adds: `use_mup` is `false` on this checkpoint, so no
//! muP `scale_depth/sqrt(num_layers)` factor is applied (unlike some
//! MiniCPM-lineage ports that assume it always is). `scale_emb` (12.0) is
//! inactive for the same reason — the reference applies it only under muP —
//! and neither knob has an inert branch here;
//! [`MiniCpm4Config`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config)
//! rejects a `use_mup=true` checkpoint outright rather than letting this
//! layer compute a different model in silence.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

pub struct MiniCpm4Layer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: MiniCpm4Attention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: MiniCpm4Mlp<R>,
}

impl<R: Runtime<DType = DType>> MiniCpm4Layer<R> {
    /// `x: [batch, seq, hidden]` -> `[batch, seq, hidden]`.
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
