//! Constructor helpers for `LlamaBlock`.

use super::attention::LlamaAttention;
use super::block::LlamaBlock;
use super::mlp::LlamaMlp;
use super::moe::{LlamaFfn, LlamaMoeMlp};
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

    // Attention projections — bias is optional (present in GPT-NeoX, some Falcon/Qwen)
    let q_bias = if attn_vb.contains("q_proj.bias") {
        Some("q_proj.bias")
    } else {
        None
    };
    let k_bias = if attn_vb.contains("k_proj.bias") {
        Some("k_proj.bias")
    } else {
        None
    };
    let v_bias = if attn_vb.contains("v_proj.bias") {
        Some("v_proj.bias")
    } else {
        None
    };
    let o_bias = if attn_vb.contains("o_proj.bias") {
        Some("o_proj.bias")
    } else {
        None
    };
    let q_proj = attn_vb.take_maybe_quant_linear("q_proj.weight", q_bias)?;
    let k_proj = attn_vb.take_maybe_quant_linear("k_proj.weight", k_bias)?;
    let v_proj = attn_vb.take_maybe_quant_linear("v_proj.weight", v_bias)?;
    let o_proj = attn_vb.take_maybe_quant_linear("o_proj.weight", o_bias)?;

    // Optional Q/K layer norms (Command-R, Cohere)
    let q_norm = attn_vb
        .take_tensor_optional("q_norm.weight")?
        .map(|w| RmsNorm::new(w, config.rms_norm_eps as f32, false));
    let k_norm = attn_vb
        .take_tensor_optional("k_norm.weight")?
        .map(|w| RmsNorm::new(w, config.rms_norm_eps as f32, false));

    let mlp = if let Some(moe_config) = &config.moe {
        // MoE: load stacked expert weights and router
        let mut mlp_vb = layer_vb.pp("block_sparse_moe");

        // Router gate: hidden_size → num_experts
        let router_gate = mlp_vb.take_maybe_quant_linear("gate.weight", None)?;

        // Expert weights stacked as [num_experts, dim_in, dim_out]
        let gate_weights = mlp_vb.take_tensor("experts.gate_proj.weight")?;
        let up_weights = mlp_vb.take_tensor("experts.up_proj.weight")?;
        let down_weights = mlp_vb.take_tensor("experts.down_proj.weight")?;

        let mut moe_mlp = LlamaMoeMlp::new(
            gate_weights,
            up_weights,
            down_weights,
            router_gate,
            moe_config.clone(),
        );

        // Optional shared expert
        if moe_config.has_shared_expert() {
            let mut shared_vb = mlp_vb.pp("shared_expert");
            let sg = shared_vb.take_maybe_quant_linear("gate_proj.weight", None)?;
            let su = shared_vb.take_maybe_quant_linear("up_proj.weight", None)?;
            let sd = shared_vb.take_maybe_quant_linear("down_proj.weight", None)?;
            moe_mlp = moe_mlp.with_shared_expert(sg, su, sd);
        }

        LlamaFfn::Moe(Box::new(moe_mlp))
    } else {
        // Dense MLP
        let mut mlp_vb = layer_vb.pp("mlp");
        let gate_proj = mlp_vb.take_maybe_quant_linear("gate_proj.weight", None)?;
        let up_proj = mlp_vb.take_maybe_quant_linear("up_proj.weight", None)?;
        let down_proj = mlp_vb.take_maybe_quant_linear("down_proj.weight", None)?;
        LlamaFfn::Dense(Box::new(LlamaMlp {
            gate_proj,
            up_proj,
            down_proj,
        }))
    };

    let use_alibi = config.attention.as_ref().is_some_and(|a| a.use_alibi);

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
            q_norm,
            k_norm,
            use_alibi,
        },
        post_attention_layernorm: RmsNorm::new(
            layer_vb.take_tensor("post_attention_layernorm.weight")?,
            config.rms_norm_eps as f32,
            false,
        ),
        mlp,
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
    let use_alibi = config.attention.as_ref().is_some_and(|a| a.use_alibi);
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
            q_norm: None,
            k_norm: None,
            use_alibi,
        },
        post_attention_layernorm: RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        ),
        mlp: LlamaFfn::Dense(Box::new(LlamaMlp {
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
        })),
    }
}
