//! `VitsEncoderLayer`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsEncoderLayer`.
//!
//! Self-attention, residual + `LayerNorm`, then [`VitsFeedForward`], residual
//! and a second `LayerNorm`. HF's `nn.Dropout` calls are training-only no-ops
//! here (inference-only module).

use crate::error::{Error, Result};
use crate::model::audio::vits::attention::VitsAttention;
use crate::model::audio::vits::feed_forward::VitsFeedForward;
use crate::model::config::VitsConfig;
use crate::nn::{LayerNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ScalarOps, ShapeOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsEncoderLayer<R: Runtime> {
    attention: VitsAttention<R>,
    layer_norm: LayerNorm<R>,
    feed_forward: VitsFeedForward<R>,
    final_layer_norm: LayerNorm<R>,
}

impl<R: Runtime<DType = DType>> VitsEncoderLayer<R> {
    /// Reads `attention.*`, `layer_norm.*`, `feed_forward.*`,
    /// `final_layer_norm.*` under `vb`'s current prefix.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &VitsConfig) -> Result<Self> {
        let mut attn_vb = vb.pp("attention");
        let attention = VitsAttention::from_varbuilder(
            &mut attn_vb,
            config.hidden_size,
            config.num_attention_heads,
            config.attention_window_size(),
            config.use_bias,
        )?;
        drop(attn_vb);

        let mut ln_vb = vb.pp("layer_norm");
        let ln_weight = ln_vb.take_tensor("weight")?;
        let ln_bias = ln_vb.take_tensor("bias")?;
        drop(ln_vb);
        let layer_norm = LayerNorm::new(ln_weight, ln_bias, config.layer_norm_eps as f32, false);

        let mut ff_vb = vb.pp("feed_forward");
        let feed_forward = VitsFeedForward::from_varbuilder(
            &mut ff_vb,
            &config.hidden_act,
            config.ffn_kernel_size,
        )?;
        drop(ff_vb);

        let mut fln_vb = vb.pp("final_layer_norm");
        let fln_weight = fln_vb.take_tensor("weight")?;
        let fln_bias = fln_vb.take_tensor("bias")?;
        drop(fln_vb);
        let final_layer_norm =
            LayerNorm::new(fln_weight, fln_bias, config.layer_norm_eps as f32, false);

        Ok(Self {
            attention,
            layer_norm,
            feed_forward,
            final_layer_norm,
        })
    }

    /// `hidden_states`: `[batch, seq, hidden]`. `attention_mask`, if given, is
    /// the additive `[batch, 1, seq, seq]` form already expanded by the
    /// caller ([`crate::model::audio::vits::encoder::VitsEncoder`]).
    /// `padding_mask`: `[batch, seq, 1]`.
    pub fn forward<C>(
        &self,
        client: &C,
        hidden_states: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
        padding_mask: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let attn_out = self
            .attention
            .forward(client, hidden_states, attention_mask)?;
        let residual = client.add(hidden_states, &attn_out).map_err(Error::Numr)?;
        let normed = self
            .layer_norm
            .forward(client, &Var::new(residual, false))?;
        let normed = normed.tensor().clone();

        let ff_out = self.feed_forward.forward(client, &normed, padding_mask)?;
        let residual = client.add(&normed, &ff_out).map_err(Error::Numr)?;
        let out = self
            .final_layer_norm
            .forward(client, &Var::new(residual, false))?;
        Ok(out.tensor().clone())
    }
}
