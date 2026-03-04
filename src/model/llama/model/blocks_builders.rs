//! Constructor helpers for `LlamaBlock`.

use super::attention::LlamaAttention;
use super::block::LlamaBlock;
use super::mlp::LlamaMlp;
use crate::error::Result;
use crate::model::config::ModelConfig;
use crate::nn::{Linear, MaybeQuantLinear, RmsNorm};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Build a LlamaBlock for a given layer from a VarBuilder.
pub fn build_block_from_varbuilder<R: Runtime<DType = DType>>(
    layer_vb: &mut crate::nn::VarBuilder<R>,
    config: &ModelConfig,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<LlamaBlock<R>> {
    let mut attn_vb = layer_vb.pp("self_attn");
    let q_proj = attn_vb.take_maybe_quant_linear("q_proj.weight", None)?;
    let k_proj = attn_vb.take_maybe_quant_linear("k_proj.weight", None)?;
    let v_proj = attn_vb.take_maybe_quant_linear("v_proj.weight", None)?;
    let o_proj = attn_vb.take_maybe_quant_linear("o_proj.weight", None)?;

    let mut mlp_vb = layer_vb.pp("mlp");
    let gate_proj = mlp_vb.take_maybe_quant_linear("gate_proj.weight", None)?;
    let up_proj = mlp_vb.take_maybe_quant_linear("up_proj.weight", None)?;
    let down_proj = mlp_vb.take_maybe_quant_linear("down_proj.weight", None)?;

    Ok(LlamaBlock {
        input_layernorm: RmsNorm::new(
            layer_vb.take_tensor("input_layernorm.weight")?,
            config.rms_norm_eps as f32,
            false,
        ),
        self_attn: LlamaAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        },
        post_attention_layernorm: RmsNorm::new(
            layer_vb.take_tensor("post_attention_layernorm.weight")?,
            config.rms_norm_eps as f32,
            false,
        ),
        mlp: LlamaMlp {
            gate_proj,
            up_proj,
            down_proj,
        },
    })
}

/// Build a LlamaBlock initialized with zeros/ones for a given device.
pub fn build_block_from_config<R: Runtime<DType = DType>>(
    config: &ModelConfig,
    device: &R::Device,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    dt: numr::dtype::DType,
) -> LlamaBlock<R> {
    let hidden = config.hidden_size;
    LlamaBlock {
        input_layernorm: RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        ),
        self_attn: LlamaAttention {
            q_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[num_heads * head_dim, hidden], dt, device),
                None,
                true,
            )),
            k_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[num_kv_heads * head_dim, hidden], dt, device),
                None,
                true,
            )),
            v_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[num_kv_heads * head_dim, hidden], dt, device),
                None,
                true,
            )),
            o_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[hidden, num_heads * head_dim], dt, device),
                None,
                true,
            )),
            num_heads,
            num_kv_heads,
            head_dim,
        },
        post_attention_layernorm: RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        ),
        mlp: LlamaMlp {
            gate_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[intermediate, hidden], dt, device),
                None,
                true,
            )),
            up_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[intermediate, hidden], dt, device),
                None,
                true,
            )),
            down_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[hidden, intermediate], dt, device),
                None,
                true,
            )),
        },
    }
}
