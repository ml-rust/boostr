//! One layer of a [`HybridModel`](super::HybridModel): the SSM, dense
//! attention or MLA block that `model.layers.<i>` holds.

use super::super::blocks::{AttentionBlock, FeedForward, MlaBlock, SsmBlock};
use super::build::HybridBlock;
use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::model::mamba::mamba2::Mamba2Config;
use crate::nn::{Linear, Mla, MlaConfig, RmsNorm, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// The shapes every layer of one model shares.
pub(super) struct LayerPlan<'a> {
    pub(super) config: &'a UniversalConfig,
    /// `Some` when the model has an SSM layer.
    pub(super) mamba: Option<&'a Mamba2Config>,
    /// `Some` when the attention layers are MLA.
    pub(super) mla: Option<&'a MlaConfig>,
    /// The dtype a quantized `kv_b_proj` dequantizes to.
    pub(super) dtype: DType,
}

impl LayerPlan<'_> {
    /// The block of layer `index`, read from `layer_vb` (`model.layers.<index>`).
    ///
    /// # Errors
    ///
    /// A tensor the block reads is missing, or the config lacks the section
    /// the layer's role needs.
    pub(super) fn build<R: Runtime<DType = DType>>(
        &self,
        layer_vb: &mut VarBuilder<R>,
        index: usize,
    ) -> Result<HybridBlock<R>>
    where
        R::Client: DequantOps<R>,
    {
        let eps = self.config.rms_norm_eps as f32;
        if self.config.is_ssm_layer(index) {
            let mamba = self.mamba.ok_or_else(|| Error::ModelError {
                reason: format!("layer {index} is an SSM layer, but the config has no ssm section"),
            })?;
            return Ok(HybridBlock::Ssm(Box::new(SsmBlock::load(
                layer_vb, mamba, eps,
            )?)));
        }
        let input_layernorm =
            RmsNorm::new(layer_vb.take_tensor("input_layernorm.weight")?, eps, false);
        let post_attention_layernorm = RmsNorm::new(
            layer_vb.take_tensor("post_attention_layernorm.weight")?,
            eps,
            false,
        );
        if let Some(mla) = self.mla {
            let attn =
                Mla::from_varbuilder(&mut layer_vb.pp("self_attn"), mla)?.absorb(self.dtype)?;
            return Ok(HybridBlock::Mla(Box::new(MlaBlock {
                input_layernorm,
                attn,
                post_attention_layernorm,
                ffn: FeedForward::load(layer_vb, self.config)?,
            })));
        }
        let attn_config = self
            .config
            .attention
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "layer {index} is an attention layer, but the config has no attention section"
                ),
            })?;
        let mut attn_vb = layer_vb.pp("self_attn");
        let q_proj = Linear::new(attn_vb.take_tensor("q_proj.weight")?, None, false);
        let k_proj = Linear::new(attn_vb.take_tensor("k_proj.weight")?, None, false);
        let v_proj = Linear::new(attn_vb.take_tensor("v_proj.weight")?, None, false);
        let o_proj = Linear::new(attn_vb.take_tensor("o_proj.weight")?, None, false);
        Ok(HybridBlock::Attention(Box::new(AttentionBlock {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_layernorm,
            ffn: FeedForward::load(layer_vb, self.config)?,
            num_heads: attn_config.num_heads,
            num_kv_heads: attn_config.kv_heads(),
            head_dim: attn_config.head_dim(self.config.hidden_size),
            use_alibi: attn_config.use_alibi,
            // `0` is the disabled sentinel; `Some(0)` is not a zero-width window.
            sliding_window: attn_config.sliding_window(),
        })))
    }
}
