//! Per-layer builders for [`Qwen35Model::from_varbuilder`](super::build::Qwen35Model):
//! one GDN layer, one attention layer, the shared norm + FFN pieces.
//!
//! `layer_vb` is scoped to `model.layers.{i}`; `layer` is `i`, needed to
//! rebuild the GGUF name the Hadamard contract is keyed on.
//!
//! `ssm_conv1d` arrives as `[qkv_dim, conv_kernel]` (GGUF `ne = [conv_kernel,
//! qkv_dim]`, reversed by the reader). `GdnBlock::new` reshapes it to the
//! `[qkv_dim, 1, conv_kernel]` layout `causal_conv1d` takes.

use super::build::{Qwen35AttentionLayer, Qwen35GdnLayer};
use super::gguf::{Attach, gguf_layer_name};
use crate::error::Result;
use crate::model::config::{GdnConfig, Qwen35AttentionConfig};
use crate::model::hybrid::{GdnBlock, GdnWeights, Qwen35AttentionBlock, Qwen35AttentionWeights};
use crate::nn::{MaybeRotatedLinear, RmsNorm, RotatedMlp, VarBuilder};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// `RmsNorm` from a `[hidden]` weight at `name`, not trainable.
pub(super) fn rms_norm<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<R>,
    name: &str,
    eps: f32,
) -> Result<RmsNorm<R>> {
    Ok(RmsNorm::new(vb.take_tensor(name)?, eps, false))
}

/// A projection at `hf_name` under `vb`, wrapped per the contract entry
/// for `blk.{layer}.{gguf_suffix}`.
fn projection<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<R>,
    attach: &Attach<'_, R>,
    layer: usize,
    hf_name: &str,
    gguf_suffix: &str,
) -> Result<MaybeRotatedLinear<R>> {
    let inner = vb.take_maybe_quant_linear(hf_name, None)?;
    attach.linear(&gguf_layer_name(layer, gguf_suffix), inner)
}

/// `mlp.{gate,up,down}_proj` as a [`RotatedMlp`].
fn build_mlp<R: Runtime<DType = DType>>(
    layer_vb: &mut VarBuilder<R>,
    attach: &Attach<'_, R>,
    layer: usize,
) -> Result<RotatedMlp<R>> {
    let mut vb = layer_vb.pp("mlp");
    let gate = projection(
        &mut vb,
        attach,
        layer,
        "gate_proj.weight",
        "ffn_gate.weight",
    )?;
    let up = projection(&mut vb, attach, layer, "up_proj.weight", "ffn_up.weight")?;
    let down = projection(
        &mut vb,
        attach,
        layer,
        "down_proj.weight",
        "ffn_down.weight",
    )?;
    RotatedMlp::new(gate, up, down)
}

/// One Gated DeltaNet layer from `model.layers.{layer}`.
pub(super) fn gdn_layer<R: Runtime<DType = DType>>(
    layer_vb: &mut VarBuilder<R>,
    attach: &Attach<'_, R>,
    layer: usize,
    cfg: &GdnConfig,
    eps: f32,
) -> Result<Qwen35GdnLayer<R>> {
    let attn_norm = rms_norm(layer_vb, "input_layernorm.weight", eps)?;
    let post_attention_norm = rms_norm(layer_vb, "post_attention_layernorm.weight", eps)?;
    let mlp = build_mlp(layer_vb, attach, layer)?;

    let mut vb = layer_vb.pp("linear_attn");
    let weights = GdnWeights {
        attn_qkv: projection(
            &mut vb,
            attach,
            layer,
            "in_proj_qkv.weight",
            "attn_qkv.weight",
        )?,
        attn_gate: projection(
            &mut vb,
            attach,
            layer,
            "in_proj_z.weight",
            "attn_gate.weight",
        )?,
        ssm_alpha: vb.take_maybe_quant_linear("alpha_proj.weight", None)?,
        ssm_beta: vb.take_maybe_quant_linear("beta_proj.weight", None)?,
        ssm_out: projection(&mut vb, attach, layer, "out_proj.weight", "ssm_out.weight")?,
        ssm_conv1d: vb.take_tensor("conv1d.weight")?,
        ssm_a: vb.take_tensor("a_neg_exp")?,
        ssm_dt_bias: vb.take_tensor("dt_bias")?,
        ssm_norm: vb.take_tensor("norm.weight")?,
    };
    let mixer = GdnBlock::new(cfg.clone(), weights)?;

    Ok(Qwen35GdnLayer {
        attn_norm,
        mixer,
        post_attention_norm,
        mlp,
    })
}

/// One gated full-attention layer from `model.layers.{layer}`.
pub(super) fn attention_layer<R: Runtime<DType = DType>>(
    layer_vb: &mut VarBuilder<R>,
    attach: &Attach<'_, R>,
    layer: usize,
    cfg: &Qwen35AttentionConfig,
    eps: f32,
) -> Result<Qwen35AttentionLayer<R>> {
    let attn_norm = rms_norm(layer_vb, "input_layernorm.weight", eps)?;
    let post_attention_norm = rms_norm(layer_vb, "post_attention_layernorm.weight", eps)?;
    let mlp = build_mlp(layer_vb, attach, layer)?;

    let mut vb = layer_vb.pp("self_attn");
    let weights = Qwen35AttentionWeights {
        attn_q: projection(&mut vb, attach, layer, "q_proj.weight", "attn_q.weight")?,
        attn_k: projection(&mut vb, attach, layer, "k_proj.weight", "attn_k.weight")?,
        attn_v: projection(&mut vb, attach, layer, "v_proj.weight", "attn_v.weight")?,
        attn_output: projection(
            &mut vb,
            attach,
            layer,
            "o_proj.weight",
            "attn_output.weight",
        )?,
        attn_q_norm: vb.take_tensor("q_norm.weight")?,
        attn_k_norm: vb.take_tensor("k_norm.weight")?,
    };
    let mixer = Qwen35AttentionBlock::new(cfg.clone(), weights)?;

    Ok(Qwen35AttentionLayer {
        attn_norm,
        mixer,
        post_attention_norm,
        mlp,
    })
}
