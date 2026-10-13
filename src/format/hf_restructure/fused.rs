//! Fused-projection splits: QKV (`query_key_value`, `qkv_proj`) into
//! `q_proj`/`k_proj`/`v_proj`, and `gate_up_proj` into `gate_proj`/`up_proj`.

use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::nn::{VarMap, Weight};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

pub(super) fn take_dense<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    name: &str,
) -> Result<Tensor<R>> {
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

pub(super) fn split_fused_attention<R: Runtime<DType = DType>>(
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

#[cfg(test)]
mod tests {
    use super::super::restructure_hf_checkpoint;
    use super::super::test_support::{config, rows, tagged};
    use crate::nn::VarMap;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

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
}
