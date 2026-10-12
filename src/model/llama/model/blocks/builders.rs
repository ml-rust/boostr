//! Constructor helpers for `LlamaBlock`.
//!
//! The checkpoint's `config.block` and its tensors decide the block shape:
//!
//! - a missing `mlp.gate_proj` gives a plain MLP
//! - a norm `bias` tensor gives a LayerNorm its bias
//! - `pre_feedforward_layernorm` gives Gemma 2 sandwich norms
//! - a `block_sparse_moe.gate` gives a sparse layer, else the layer is dense
//!
//! Partial and interleaved RoPE, Gemma norm offsets and query scales are
//! weight edits that `restructure_hf_checkpoint` makes before this runs.

use super::attention::LlamaAttention;
use super::block::LlamaBlock;
use super::mlp::LlamaMlp;
use super::moe::{LlamaFfn, LlamaMoeMlp};
use super::norm::BlockNorm;
use crate::error::{Error, Result};
use crate::model::config::{BlockConfig, ModelConfig, ResidualLayout};
use crate::nn::{Linear, MaybeQuantLinear, RmsNorm, VarBuilder};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Take `{name}.weight` and its optional bias as a linear layer.
fn take_projection<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<R>,
    name: &str,
) -> Result<MaybeQuantLinear<R>> {
    let bias_name = format!("{name}.bias");
    let bias = vb.contains(&bias_name).then_some(bias_name.as_str());
    vb.take_maybe_quant_linear(&format!("{name}.weight"), bias)
}

/// Take `{name}.weight` and an optional `{name}.bias` as a block norm.
fn take_norm<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<R>,
    name: &str,
    config: &ModelConfig,
) -> Result<BlockNorm<R>> {
    let weight = vb.take_tensor(&format!("{name}.weight"))?;
    let bias = vb.take_tensor_optional(&format!("{name}.bias"))?;
    BlockNorm::build(
        weight,
        bias,
        config.rms_norm_eps as f32,
        &config.block,
        false,
    )
}

fn take_norm_optional<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<R>,
    name: &str,
    config: &ModelConfig,
) -> Result<Option<BlockNorm<R>>> {
    if vb.contains(&format!("{name}.weight")) {
        take_norm(vb, name, config).map(Some)
    } else {
        Ok(None)
    }
}

/// The optional per-head Q/K RMSNorm.
fn take_qk_norm<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<R>,
    name: &str,
    config: &ModelConfig,
) -> Result<Option<RmsNorm<R>>> {
    Ok(vb
        .take_tensor_optional(name)?
        .map(|w| RmsNorm::new(w, config.rms_norm_eps as f32, false)))
}

/// The FFN of one layer: sparse when the layer has a router, else dense.
fn build_ffn<R: Runtime<DType = DType>>(
    layer_vb: &mut VarBuilder<R>,
    config: &ModelConfig,
) -> Result<LlamaFfn<R>> {
    let activation = config.block.mlp_activation;
    if let Some(moe_config) = &config.moe
        && layer_vb.contains("block_sparse_moe.gate.weight")
    {
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
        )
        .with_activation(activation);

        // Optional shared expert, by config or by tensor (Qwen2-MoE).
        if moe_config.has_shared_expert() || mlp_vb.contains("shared_expert.gate_proj.weight") {
            let mut shared_vb = mlp_vb.pp("shared_expert");
            let sg = shared_vb.take_maybe_quant_linear("gate_proj.weight", None)?;
            let su = shared_vb.take_maybe_quant_linear("up_proj.weight", None)?;
            let sd = shared_vb.take_maybe_quant_linear("down_proj.weight", None)?;
            moe_mlp = moe_mlp.with_shared_expert(sg, su, sd);
        }
        if let Some(gate) =
            mlp_vb.take_maybe_quant_linear_optional("shared_expert_gate.weight", None)?
        {
            moe_mlp = moe_mlp.with_shared_expert_gate(gate);
        }
        return Ok(LlamaFfn::Moe(Box::new(moe_mlp)));
    }

    let mut mlp_vb = layer_vb.pp("mlp");
    let take = |vb: &mut VarBuilder<R>, name: &str| -> Result<Option<MaybeQuantLinear<R>>> {
        let bias = format!("{name}.bias");
        let bias = vb.contains(&bias).then_some(bias);
        vb.take_maybe_quant_linear_optional(&format!("{name}.weight"), bias.as_deref())
    };
    let gate_proj = take(&mut mlp_vb, "gate_proj")?;
    let missing = |name: &str, vb: &VarBuilder<R>| Error::ModelError {
        reason: format!("{} is missing", vb.full_name(&format!("{name}.weight"))),
    };
    let up_proj = take(&mut mlp_vb, "up_proj")?.ok_or_else(|| missing("up_proj", &mlp_vb))?;
    let down_proj = take(&mut mlp_vb, "down_proj")?.ok_or_else(|| missing("down_proj", &mlp_vb))?;
    Ok(LlamaFfn::Dense(Box::new(LlamaMlp {
        gate_proj,
        up_proj,
        down_proj,
        activation,
    })))
}

/// Build the LlamaBlock of layer `layer` from a VarBuilder.
///
/// The tensors must already be in the Llama layout. For a Hugging Face
/// checkpoint of a non-Llama architecture,
/// [`restructure_hf_checkpoint`](crate::format::restructure_hf_checkpoint)
/// puts them there, and this refuses a map it did not restructure.
pub fn build_block_from_varbuilder<R: Runtime<DType = DType>>(
    layer_vb: &mut VarBuilder<R>,
    config: &ModelConfig,
    layer: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<LlamaBlock<R>> {
    let block = &config.block;
    if needs_weight_transform(block, head_dim) && !layer_vb.is_hf_restructured() {
        return Err(Error::ModelError {
            reason: format!(
                "model type '{}' needs its Q/K rows reordered, its query scale folded or its \
                 norm weights offset at load; run crate::format::restructure_hf_checkpoint \
                 on the tensors first",
                config.model_type
            ),
        });
    }
    let use_alibi = config.attention.as_ref().is_some_and(|a| a.use_alibi);

    let mut attn_vb = layer_vb.pp("self_attn");
    let q_proj = take_projection(&mut attn_vb, "q_proj")?;
    let k_proj = take_projection(&mut attn_vb, "k_proj")?;
    let v_proj = take_projection(&mut attn_vb, "v_proj")?;
    let o_proj = take_projection(&mut attn_vb, "o_proj")?;

    // Optional Q/K norms (Qwen3).
    let q_norm = take_qk_norm(&mut attn_vb, "q_norm.weight", config)?;
    let k_norm = take_qk_norm(&mut attn_vb, "k_norm.weight", config)?;

    let mlp = build_ffn(layer_vb, config)?;

    let sliding_window = if block.layer_slides(layer) {
        config.attention.as_ref().map_or(0, |a| a.sliding_window())
    } else {
        0
    };

    let input_layernorm = take_norm(layer_vb, "input_layernorm", config)?;
    let post_attention_layernorm =
        take_norm_optional(layer_vb, "post_attention_layernorm", config)?;
    let pre_feedforward_layernorm =
        take_norm_optional(layer_vb, "pre_feedforward_layernorm", config)?;
    let post_feedforward_layernorm =
        take_norm_optional(layer_vb, "post_feedforward_layernorm", config)?;
    if post_attention_layernorm.is_none() && block.residual != ResidualLayout::ParallelSharedNorm {
        return Err(Error::ModelError {
            reason: format!(
                "{} is missing; only a shared-norm parallel block runs without it",
                layer_vb.full_name("post_attention_layernorm.weight")
            ),
        });
    }

    Ok(LlamaBlock {
        input_layernorm,
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
            sliding_window,
            softcap: block.attn_logit_softcap,
            clip_qkv: block.clip_qkv,
            alibi_scale: block.alibi_bias_scale.unwrap_or(1.0),
        },
        post_attention_layernorm,
        pre_feedforward_layernorm,
        post_feedforward_layernorm,
        mlp,
        residual: block.residual,
    })
}

/// Whether `block` asks for load-time weight edits that
/// [`restructure_hf_checkpoint`](crate::format::restructure_hf_checkpoint)
/// makes: a norm offset, a query-scale fold, or a Q/K row reorder.
pub(crate) fn needs_weight_transform(block: &BlockConfig, head_dim: usize) -> bool {
    block.norm_weight_offset != 0.0
        || crate::format::hf_restructure::query_scale_fold(block, head_dim).is_some()
        || crate::format::hf_restructure::rotary_row_order(
            head_dim,
            block.rotary_dim(head_dim),
            block.rope_interleaved,
        )
        .is_some()
}

/// Build a LlamaBlock initialized with zeros/ones for a given device.
///
/// Training from scratch builds the Llama block only.
pub fn build_block_from_config<R: Runtime<DType = DType>>(
    config: &ModelConfig,
    device: &R::Device,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    dt: numr::dtype::DType,
) -> Result<LlamaBlock<R>> {
    if !config.block.is_llama() {
        return Err(Error::ModelError {
            reason: format!(
                "model type '{}' sets block options; from-scratch construction builds the \
                 Llama block only",
                config.model_type
            ),
        });
    }
    let hidden = config.hidden_size;
    let use_alibi = config.attention.as_ref().is_some_and(|a| a.use_alibi);
    let sliding_window = config.attention.as_ref().map_or(0, |a| a.sliding_window());
    let linear = |rows: usize, cols: usize| -> Result<MaybeQuantLinear<R>> {
        Ok(MaybeQuantLinear::Standard(Linear::new(
            Tensor::<R>::zeros(&[rows, cols], dt, device)?,
            None,
            true,
        )))
    };
    let norm = || -> Result<BlockNorm<R>> {
        Ok(BlockNorm::Rms(RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device)?,
            config.rms_norm_eps as f32,
            true,
        )))
    };
    Ok(LlamaBlock {
        input_layernorm: norm()?,
        self_attn: LlamaAttention {
            q_proj: linear(num_heads * head_dim, hidden)?,
            k_proj: linear(num_kv_heads * head_dim, hidden)?,
            v_proj: linear(num_kv_heads * head_dim, hidden)?,
            o_proj: linear(hidden, num_heads * head_dim)?,
            num_heads,
            num_kv_heads,
            head_dim,
            q_norm: None,
            k_norm: None,
            use_alibi,
            sliding_window,
            softcap: None,
            clip_qkv: None,
            alibi_scale: 1.0,
        },
        post_attention_layernorm: Some(norm()?),
        pre_feedforward_layernorm: None,
        post_feedforward_layernorm: None,
        mlp: LlamaFfn::Dense(Box::new(LlamaMlp::swiglu(
            linear(intermediate, hidden)?,
            linear(intermediate, hidden)?,
            linear(hidden, intermediate)?,
        ))),
        residual: ResidualLayout::Sequential,
    })
}
