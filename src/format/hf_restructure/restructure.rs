//! The restructure pipeline: rename, split fused projections, stack MoE
//! experts, then make the block's weight edits.
//!
//! The Llama path reads separate `q_proj`/`k_proj`/`v_proj`, separate
//! `gate_proj`/`up_proj`, and MoE experts stacked as `[experts, in, out]`.
//! Several architectures store them otherwise:
//!
//! | Architecture | Stored as | Row order |
//! | --- | --- | --- |
//! | GPT-NeoX, Falcon (`multi_query: false`) | `query_key_value` | `[heads][q,k,v][head_dim]` |
//! | Falcon (`new_decoder_architecture`) | `query_key_value` | `[kv_heads][q.., k, v][head_dim]` |
//! | Falcon (`multi_query`), DBRX, Phi-3 | `query_key_value` / `qkv_proj` | `[q][k][v]` |
//! | Phi-3 | `gate_up_proj` | `[gate][up]` |
//! | Mixtral, Qwen2-MoE | one `[out, in]` tensor per expert | |
//! | DBRX | `w1`, `v1`, `w2` fused over experts, `[experts * ffn, hidden]` | |
//!
//! [`restructure_hf_checkpoint`] renames with
//! [`normalize_hf_name`], then
//! splits and stacks. It also makes the weight edits `config.block` asks for:
//!
//! - Gemma stores each norm weight as `w - 1`. The offset is added back.
//! - A query scale other than `1/sqrt(head_dim)` folds into `q_proj`.
//! - Partial (GPT-NeoX) and interleaved (Cohere) RoPE reorder each head's Q
//!   and K rows, so the split-half kernel rotates the right pairs. The row
//!   order comes from `rotary_row_order`.
//!
//! A Llama-layout checkpoint passes through unchanged.

use super::block_edits::apply_block_edits;
use super::experts::{split_dbrx_experts, stack_per_expert};
use super::fused::split_fused_attention;
use crate::error::{Error, Result};
use crate::format::safetensors_name_map::normalize_hf_name;
use crate::model::config::UniversalConfig;
use crate::nn::VarMap;
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, ShapeOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;

/// Rename a Hugging Face checkpoint's tensors to the Llama names, split
/// fused projections, stack MoE experts, and make the block's weight edits.
///
/// Run it once per map. It marks the map, and a second run is an error.
///
/// # Errors
///
/// A fused tensor whose row count does not fit the config, a quantized
/// fused tensor, or a gap in the expert ids.
pub fn restructure_hf_checkpoint<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    config: &UniversalConfig,
    _device: &R::Device,
) -> Result<()>
where
    R::Client: ShapeOps<R> + BinaryOps<R> + IndexingOps<R> + UnaryOps<R> + TypeConversionOps<R>,
{
    if map.is_hf_restructured() {
        return Err(Error::ModelError {
            reason: "restructure_hf_checkpoint already ran on this map; a second run would \
                     apply its weight edits twice"
                .into(),
        });
    }
    let model_type = config.model_type.as_str();
    if model_type == "qwen35" {
        return crate::format::hf_qwen35::restructure(map, config);
    }
    let names: Vec<String> = map.names().map(str::to_string).collect();
    for name in names {
        let mapped = normalize_hf_name(model_type, &name);
        if mapped != name
            && let Some(weight) = map.remove(&name)
        {
            map.insert_weight(mapped, weight);
        }
    }
    split_fused_attention(map, config)?;
    if model_type == "dbrx" {
        split_dbrx_experts(map, config)?;
    }
    stack_per_expert(map)?;
    apply_block_edits(map, config)?;
    map.mark_hf_restructured();
    Ok(())
}
