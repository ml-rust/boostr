//! Put a Hugging Face SafeTensors checkpoint into the Llama tensor layout.
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

use crate::error::{Error, Result};
use crate::format::safetensors_name_map::normalize_hf_name;
use crate::model::config::{BlockConfig, UniversalConfig};
use crate::nn::{VarMap, Weight};
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::BTreeMap;

/// Row order of a fused QKV projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QkvLayout {
    /// `[q (heads * d)][k (kv * d)][v (kv * d)]`.
    Contiguous,
    /// `[kv groups][heads/kv queries, 1 key, 1 value][d]`. GPT-NeoX and old
    /// Falcon are the case `kv == heads`.
    Grouped,
}

fn qkv_layout(config: &UniversalConfig) -> QkvLayout {
    match config.model_type.as_str() {
        "gpt_neox" => QkvLayout::Grouped,
        "falcon" => {
            let attn = config.attention.as_ref();
            let kv = attn.map_or(0, |a| a.kv_heads());
            let heads = attn.map_or(0, |a| a.num_heads);
            // Multi-query Falcon stores `[q][k][v]`. Every other Falcon
            // groups the rows per KV head.
            if kv == 1 && heads > 1 {
                QkvLayout::Contiguous
            } else {
                QkvLayout::Grouped
            }
        }
        _ => QkvLayout::Contiguous,
    }
}

fn take_dense<R: Runtime<DType = DType>>(map: &mut VarMap<R>, name: &str) -> Result<Tensor<R>> {
    match map.remove(name) {
        Some(Weight::Standard(t)) => Ok(t),
        Some(_) => Err(Error::ModelError {
            reason: format!(
                "{name} is quantized; a fused projection splits only from F32, F16 or BF16"
            ),
        }),
        None => Err(Error::ModelError {
            reason: format!("{name} is missing"),
        }),
    }
}

/// `[rows, rest..]` → `[a, b, c, rest..]`, then rows `start..start+len` of
/// axis 1, back to `[a * len * c, rest..]`.
fn take_rows<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
    groups: (usize, usize, usize),
    start: usize,
    len: usize,
) -> Result<Tensor<R>> {
    let (a, b, c) = groups;
    let rest: Vec<usize> = t.shape()[1..].to_vec();
    let mut grouped = vec![a, b, c];
    grouped.extend_from_slice(&rest);
    let part = t
        .contiguous()?
        .reshape(&grouped)?
        .narrow(1, start, len)?
        .contiguous()?;
    let mut flat = vec![a * len * c];
    flat.extend_from_slice(&rest);
    Ok(part.reshape(&flat)?)
}

/// Split one fused QKV tensor (`weight` `[rows, in]` or `bias` `[rows]`).
fn split_qkv<R: Runtime<DType = DType>>(
    fused: &Tensor<R>,
    layout: QkvLayout,
    heads: usize,
    kv: usize,
    d: usize,
    name: &str,
) -> Result<[Tensor<R>; 3]> {
    let rows = fused.shape().first().copied().unwrap_or(0);
    let want = (heads + 2 * kv) * d;
    if rows != want {
        return Err(Error::ModelError {
            reason: format!(
                "{name} has {rows} rows; {heads} query heads and {kv} KV heads of size {d} \
                 need {want}"
            ),
        });
    }
    Ok(match layout {
        QkvLayout::Contiguous => [
            take_rows(fused, (1, heads + 2 * kv, d), 0, heads)?,
            take_rows(fused, (1, heads + 2 * kv, d), heads, kv)?,
            take_rows(fused, (1, heads + 2 * kv, d), heads + kv, kv)?,
        ],
        QkvLayout::Grouped => {
            let g = heads / kv;
            [
                take_rows(fused, (kv, g + 2, d), 0, g)?,
                take_rows(fused, (kv, g + 2, d), g, 1)?,
                take_rows(fused, (kv, g + 2, d), g + 1, 1)?,
            ]
        }
    })
}

fn split_fused_attention<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    config: &UniversalConfig,
) -> Result<()> {
    let Some(attn) = &config.attention else {
        return Ok(());
    };
    let (heads, kv) = (attn.num_heads, attn.kv_heads());
    let d = attn.head_dim(config.hidden_size);
    let layout = qkv_layout(config);
    for layer in 0..config.num_layers {
        let prefix = format!("model.layers.{layer}.self_attn");
        for fused in ["query_key_value", "qkv_proj"] {
            for suffix in ["weight", "bias"] {
                let name = format!("{prefix}.{fused}.{suffix}");
                if !map.contains(&name) {
                    continue;
                }
                let t = take_dense(map, &name)?;
                let [q, k, v] = split_qkv(&t, layout, heads, kv, d, &name)?;
                map.insert(format!("{prefix}.q_proj.{suffix}"), q);
                map.insert(format!("{prefix}.k_proj.{suffix}"), k);
                map.insert(format!("{prefix}.v_proj.{suffix}"), v);
            }
        }
        for suffix in ["weight", "bias"] {
            let name = format!("model.layers.{layer}.mlp.gate_up_proj.{suffix}");
            if !map.contains(&name) {
                continue;
            }
            let t = take_dense(map, &name)?;
            let rows = t.shape()[0];
            if rows % 2 != 0 {
                return Err(Error::ModelError {
                    reason: format!("{name} has an odd row count {rows}; it holds gate then up"),
                });
            }
            let half = rows / 2;
            map.insert(
                format!("model.layers.{layer}.mlp.gate_proj.{suffix}"),
                t.narrow(0, 0, half)?.contiguous()?,
            );
            map.insert(
                format!("model.layers.{layer}.mlp.up_proj.{suffix}"),
                t.narrow(0, half, half)?.contiguous()?,
            );
        }
    }
    Ok(())
}

/// `[out, in]` → `[in, out]`, contiguous.
fn transposed<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Result<Tensor<R>> {
    Ok(t.transpose(0, 1)?.contiguous()?)
}

/// Stack per-expert `[out, in]` projections as `[experts, in, out]`.
fn stack_per_expert<R: Runtime<DType = DType>>(map: &mut VarMap<R>) -> Result<()>
where
    R::Client: ShapeOps<R>,
{
    let marker = ".block_sparse_moe.experts.";
    // stacked name → expert id → stored name
    let mut groups: BTreeMap<String, BTreeMap<usize, String>> = BTreeMap::new();
    for name in map.names() {
        let Some(pos) = name.find(marker) else {
            continue;
        };
        let after = &name[pos + marker.len()..];
        let Some((id, proj)) = after.split_once('.') else {
            continue;
        };
        let Ok(id) = id.parse::<usize>() else {
            continue;
        };
        groups
            .entry(format!("{}{marker}{proj}", &name[..pos]))
            .or_default()
            .insert(id, name.to_string());
    }
    for (stacked, experts) in groups {
        if let Some((&last, _)) = experts.iter().next_back()
            && last + 1 != experts.len()
        {
            return Err(Error::ModelError {
                reason: format!(
                    "{stacked}: expert ids are not 0..{}; expert {last} is the highest",
                    experts.len()
                ),
            });
        }
        let mut parts = Vec::with_capacity(experts.len());
        for name in experts.values() {
            parts.push(transposed(&take_dense(map, name)?)?);
        }
        let refs: Vec<&Tensor<R>> = parts.iter().collect();
        map.insert(stacked, Tensor::stack(&refs, 0)?);
    }
    Ok(())
}

/// DBRX `w1`, `v1`, `w2`, each `[experts * ffn, hidden]`, as stacked
/// `gate_proj`, `up_proj` `[experts, hidden, ffn]` and `down_proj`
/// `[experts, ffn, hidden]`.
fn split_dbrx_experts<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    config: &UniversalConfig,
) -> Result<()> {
    let Some(moe) = &config.moe else {
        return Ok(());
    };
    let e = moe.num_experts;
    for layer in 0..config.num_layers {
        let base = format!("model.layers.{layer}.block_sparse_moe.experts");
        for (part, proj, transpose) in [
            ("dbrx_w1", "gate_proj", true),
            ("dbrx_v1", "up_proj", true),
            ("dbrx_w2", "down_proj", false),
        ] {
            let name = format!("{base}.{part}");
            if !map.contains(&name) {
                continue;
            }
            let t = take_dense(map, &name)?;
            let (rows, hidden) = (t.shape()[0], t.shape()[1]);
            if rows % e != 0 {
                return Err(Error::ModelError {
                    reason: format!("{name} has {rows} rows, not a multiple of {e} experts"),
                });
            }
            let per = t.contiguous()?.reshape(&[e, rows / e, hidden])?;
            let stacked = if transpose {
                per.permute(&[0, 2, 1])?.contiguous()?
            } else {
                per
            };
            map.insert(format!("{base}.{proj}.weight"), stacked);
        }
    }
    Ok(())
}

/// The source dim of each target dim of one head, so that the rotated pairs
/// sit at `(j, j + head_dim/2)` with the rotated dims first in each half.
///
/// - split-half partial RoPE (GPT-NeoX): pairs `(j, j + rotary_dim/2)`
/// - interleaved RoPE (Cohere): pairs `(2j, 2j + 1)`
///
/// Unrotated dims fill the rest of each half in order. Attention scores are
/// sums over dims, so the same reorder of Q and K leaves them unchanged.
/// `None` when the order is already the identity.
pub(crate) fn rotary_row_order(
    head_dim: usize,
    rotary_dim: usize,
    interleaved: bool,
) -> Option<Vec<usize>> {
    if rotary_dim >= head_dim && !interleaved {
        return None;
    }
    let half_rot = rotary_dim / 2;
    let pass = head_dim - rotary_dim;
    let (first, second): (Vec<usize>, Vec<usize>) = (0..half_rot)
        .map(|j| {
            if interleaved {
                (2 * j, 2 * j + 1)
            } else {
                (j, j + half_rot)
            }
        })
        .unzip();
    let mut order = first;
    order.extend(rotary_dim..rotary_dim + pass / 2);
    order.extend(second);
    order.extend(rotary_dim + pass / 2..head_dim);
    Some(order)
}

/// The query-scale fold: `scale * sqrt(head_dim)`, or `None` when it is 1.
pub(crate) fn query_scale_fold(block: &BlockConfig, head_dim: usize) -> Option<f32> {
    let scale = block.attention_scale?;
    let fold = scale * (head_dim as f32).sqrt();
    ((fold - 1.0).abs() > f32::EPSILON).then_some(fold)
}

/// `t` with each head's rows (`[heads * head_dim, ..]`) put in `order`.
fn reorder_head_rows<R: Runtime<DType = DType>>(t: &Tensor<R>, order: &[usize]) -> Result<Tensor<R>>
where
    R::Client: IndexingOps<R>,
{
    let d = order.len();
    let rows = t.shape().first().copied().unwrap_or(0);
    if d == 0 || rows % d != 0 {
        return Err(Error::ModelError {
            reason: format!("{rows} rows do not split into heads of {d}"),
        });
    }
    let index: Vec<i64> = (0..rows / d)
        .flat_map(|h| order.iter().map(move |&j| (h * d + j) as i64))
        .collect();
    let index = Tensor::<R>::from_slice(&index, &[index.len()], t.device())?;
    Ok(R::default_client(t.device()).index_select(t, 0, &index)?)
}

/// `t * factor` or `t + offset`, with the scalar in `t`'s dtype.
fn scalar_op<R: Runtime<DType = DType>>(t: &Tensor<R>, value: f32, add: bool) -> Result<Tensor<R>>
where
    R::Client: BinaryOps<R>,
{
    let s = Tensor::<R>::full_scalar(&[1], t.dtype(), f64::from(value), t.device())?;
    let client = R::default_client(t.device());
    Ok(if add {
        client.add(t, &s)?
    } else {
        client.mul(t, &s)?
    })
}

/// Apply `f` to the dense tensor `name`, when the map holds it.
fn edit_tensor<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    name: &str,
    f: impl FnOnce(&Tensor<R>) -> Result<Tensor<R>>,
) -> Result<()> {
    if !map.contains(name) {
        return Ok(());
    }
    let t = take_dense(map, name)?;
    map.insert(name.to_string(), f(&t)?);
    Ok(())
}

/// The weight edits `config.block` asks for. See the module docs.
fn apply_block_edits<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    config: &UniversalConfig,
) -> Result<()>
where
    R::Client: BinaryOps<R> + IndexingOps<R>,
{
    let block = &config.block;
    if block.norm_weight_offset != 0.0 {
        let norms: Vec<String> = map
            .names()
            .filter(|n| n.ends_with("norm.weight"))
            .map(str::to_string)
            .collect();
        for name in norms {
            edit_tensor(map, &name, |t| scalar_op(t, block.norm_weight_offset, true))?;
        }
    }
    let Some(attn) = &config.attention else {
        return Ok(());
    };
    let d = attn.head_dim(config.hidden_size);
    let order = if attn.use_alibi {
        None
    } else {
        rotary_row_order(d, block.rotary_dim(d), block.rope_interleaved)
    };
    let fold = query_scale_fold(block, d);
    for layer in 0..config.num_layers {
        let p = format!("model.layers.{layer}.self_attn");
        if let Some(order) = &order {
            for name in [
                format!("{p}.q_proj.weight"),
                format!("{p}.q_proj.bias"),
                format!("{p}.k_proj.weight"),
                format!("{p}.k_proj.bias"),
                format!("{p}.q_norm.weight"),
                format!("{p}.k_norm.weight"),
            ] {
                edit_tensor(map, &name, |t| reorder_head_rows(t, order))?;
            }
        }
        if let Some(f) = fold {
            // A Q norm would undo a fold into q_proj, so it takes the fold.
            let targets = if map.contains(&format!("{p}.q_norm.weight")) {
                vec![format!("{p}.q_norm.weight")]
            } else {
                vec![format!("{p}.q_proj.weight"), format!("{p}.q_proj.bias")]
            };
            for name in targets {
                edit_tensor(map, &name, |t| scalar_op(t, f, false))?;
            }
        }
    }
    Ok(())
}

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
    R::Client: ShapeOps<R> + BinaryOps<R> + IndexingOps<R> + UnaryOps<R>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn config(json: &str) -> UniversalConfig {
        let hf: crate::model::config::HuggingFaceConfig = serde_json::from_str(json).unwrap();
        hf.to_universal().unwrap()
    }

    fn rows(t: &Tensor<CpuRuntime>) -> Vec<f32> {
        t.contiguous().unwrap().to_vec()
    }

    /// Row `r` of a `[n, 1]` tensor holds `r`.
    fn tagged(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
        let v: Vec<f32> = (0..n).map(|r| r as f32).collect();
        Tensor::from_slice(&v, &[n, 1], device).unwrap()
    }

    #[test]
    fn gpt_neox_qkv_rows_split_per_head() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "gpt_neox", "vocab_size": 4, "hidden_size": 4,
                "num_hidden_layers": 1, "num_attention_heads": 2}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "gpt_neox.layers.0.attention.query_key_value.weight".into(),
            tagged(12, &device),
        );
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        let q = map
            .get_tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let k = map
            .get_tensor("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        let v = map
            .get_tensor("model.layers.0.self_attn.v_proj.weight")
            .unwrap();
        // Head h owns rows 6h..6h+6 as [q q k k v v].
        assert_eq!(rows(q), vec![0.0, 1.0, 6.0, 7.0]);
        assert_eq!(rows(k), vec![2.0, 3.0, 8.0, 9.0]);
        assert_eq!(rows(v), vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn phi3_qkv_and_gate_up_split_contiguously() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "phi3", "vocab_size": 4, "hidden_size": 4,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "model.layers.0.self_attn.qkv_proj.weight".into(),
            tagged(8, &device),
        );
        map.insert(
            "model.layers.0.mlp.gate_up_proj.weight".into(),
            tagged(6, &device),
        );
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        let get = |n: &str| rows(map.get_tensor(n).unwrap());
        assert_eq!(
            get("model.layers.0.self_attn.q_proj.weight"),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            get("model.layers.0.self_attn.k_proj.weight"),
            vec![4.0, 5.0]
        );
        assert_eq!(
            get("model.layers.0.self_attn.v_proj.weight"),
            vec![6.0, 7.0]
        );
        assert_eq!(
            get("model.layers.0.mlp.gate_proj.weight"),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(
            get("model.layers.0.mlp.up_proj.weight"),
            vec![3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn falcon_new_decoder_qkv_groups_per_kv_head() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "falcon", "vocab_size": 4, "hidden_size": 4,
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_kv_heads": 2,
                "new_decoder_architecture": true}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        // 2 groups of [q q k v], head_dim 1.
        map.insert(
            "transformer.h.0.self_attention.query_key_value.weight".into(),
            tagged(8, &device),
        );
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        let get = |n: &str| rows(map.get_tensor(n).unwrap());
        assert_eq!(
            get("model.layers.0.self_attn.q_proj.weight"),
            vec![0.0, 1.0, 4.0, 5.0]
        );
        assert_eq!(
            get("model.layers.0.self_attn.k_proj.weight"),
            vec![2.0, 6.0]
        );
        assert_eq!(
            get("model.layers.0.self_attn.v_proj.weight"),
            vec![3.0, 7.0]
        );
    }

    #[test]
    fn mixtral_experts_stack_transposed() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "mixtral", "vocab_size": 4, "hidden_size": 2,
                "num_hidden_layers": 1, "num_attention_heads": 1,
                "num_local_experts": 2, "num_experts_per_tok": 1}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        for e in 0..2 {
            // [out=3, in=2]
            let v: Vec<f32> = (0..6).map(|i| (e * 10 + i) as f32).collect();
            map.insert(
                format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight"),
                Tensor::from_slice(&v, &[3, 2], &device).unwrap(),
            );
        }
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        let s = map
            .get_tensor("model.layers.0.block_sparse_moe.experts.gate_proj.weight")
            .unwrap();
        assert_eq!(s.shape(), &[2, 2, 3]);
        assert_eq!(
            rows(s),
            vec![
                0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 10.0, 12.0, 14.0, 11.0, 13.0, 15.0
            ]
        );
    }

    #[test]
    fn dbrx_fused_experts_split_per_expert() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "dbrx", "vocab_size": 4, "d_model": 2, "n_heads": 1,
                "n_layers": 1, "max_seq_len": 8,
                "attn_config": {"kv_n_heads": 1},
                "ffn_config": {"ffn_hidden_size": 3, "moe_num_experts": 2, "moe_top_k": 1}}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        for part in ["w1", "w2"] {
            map.insert(
                format!("transformer.blocks.0.ffn.experts.mlp.{part}"),
                Tensor::from_slice(&v, &[6, 2], &device).unwrap(),
            );
        }
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        let gate = map
            .get_tensor("model.layers.0.block_sparse_moe.experts.gate_proj.weight")
            .unwrap();
        assert_eq!(gate.shape(), &[2, 2, 3]);
        assert_eq!(&rows(gate)[..6], &[0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        let down = map
            .get_tensor("model.layers.0.block_sparse_moe.experts.down_proj.weight")
            .unwrap();
        assert_eq!(down.shape(), &[2, 3, 2]);
        assert_eq!(&rows(down)[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn wrong_fused_row_count_is_an_error() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "phi3", "vocab_size": 4, "hidden_size": 4,
                "num_hidden_layers": 1, "num_attention_heads": 2}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "model.layers.0.self_attn.qkv_proj.weight".into(),
            tagged(7, &device),
        );
        let err = restructure_hf_checkpoint(&mut map, &c, &device).unwrap_err();
        assert!(err.to_string().contains("qkv_proj"), "{err}");
    }

    #[test]
    fn full_split_half_rope_keeps_the_rows() {
        assert_eq!(rotary_row_order(8, 8, false), None);
    }

    /// GPT-NeoX head_dim 8, rotary 4: pairs (0,2), (1,3); dims 4..8 pass.
    #[test]
    fn partial_rope_puts_rotated_pairs_first_in_each_half() {
        assert_eq!(
            rotary_row_order(8, 4, false),
            Some(vec![0, 1, 4, 5, 2, 3, 6, 7])
        );
    }

    /// Cohere: pairs (0,1), (2,3), ... become (j, j + 4).
    #[test]
    fn interleaved_rope_deinterleaves_the_pairs() {
        assert_eq!(
            rotary_row_order(8, 8, true),
            Some(vec![0, 2, 4, 6, 1, 3, 5, 7])
        );
    }

    #[test]
    fn gemma_norms_gain_one_and_queries_take_the_scale() {
        let device = CpuDevice::new();
        let c = config(
            r#"{"model_type": "gemma2", "vocab_size": 4, "hidden_size": 4,
                "num_hidden_layers": 1, "num_attention_heads": 1, "head_dim": 4,
                "query_pre_attn_scalar": 1.0}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "model.layers.0.input_layernorm.weight".into(),
            Tensor::from_slice(&[0.5f32, -1.0, 0.0, 2.0], &[4], &device).unwrap(),
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            tagged(4, &device),
        );
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        assert!(map.is_hf_restructured());
        assert_eq!(
            rows(
                map.get_tensor("model.layers.0.input_layernorm.weight")
                    .unwrap()
            ),
            vec![1.5, 0.0, 1.0, 3.0]
        );
        // scale 1 with head_dim 4 folds 1 * sqrt(4) = 2 into q_proj.
        assert_eq!(
            rows(
                map.get_tensor("model.layers.0.self_attn.q_proj.weight")
                    .unwrap()
            ),
            vec![0.0, 2.0, 4.0, 6.0]
        );
        assert!(restructure_hf_checkpoint(&mut map, &c, &device).is_err());
    }

    #[test]
    fn gpt_neox_rotary_rows_reorder_q_and_k_only() {
        let device = CpuDevice::new();
        // head_dim 8, rotary_pct 0.5: rotary 4.
        let c = config(
            r#"{"model_type": "gpt_neox", "vocab_size": 4, "hidden_size": 8,
                "num_hidden_layers": 1, "num_attention_heads": 1, "rotary_pct": 0.5}"#,
        );
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "gpt_neox.layers.0.attention.query_key_value.weight".into(),
            tagged(24, &device),
        );
        restructure_hf_checkpoint(&mut map, &c, &device).unwrap();
        let get = |n: &str| rows(map.get_tensor(n).unwrap());
        assert_eq!(
            get("model.layers.0.self_attn.q_proj.weight"),
            vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]
        );
        assert_eq!(
            get("model.layers.0.self_attn.k_proj.weight"),
            vec![8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0, 15.0]
        );
        assert_eq!(
            get("model.layers.0.self_attn.v_proj.weight"),
            (16..24).map(|r| r as f32).collect::<Vec<_>>()
        );
    }
}
