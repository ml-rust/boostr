//! MoE expert stacking: per-expert projections (Mixtral, Qwen2-MoE) and
//! DBRX's experts-fused `w1`/`v1`/`w2`, as stacked `[experts, ..]` tensors.

use super::fused::take_dense;
use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::nn::VarMap;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::BTreeMap;

/// `[out, in]` → `[in, out]`, contiguous.
fn transposed<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Result<Tensor<R>> {
    Ok(t.transpose(0, 1)?.contiguous()?)
}

/// Stack per-expert `[out, in]` projections as `[experts, in, out]`.
pub(super) fn stack_per_expert<R: Runtime<DType = DType>>(map: &mut VarMap<R>) -> Result<()>
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
pub(super) fn split_dbrx_experts<R: Runtime<DType = DType>>(
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

#[cfg(test)]
mod tests {
    use super::super::restructure_hf_checkpoint;
    use super::super::test_support::{config, rows};
    use crate::nn::VarMap;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

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
}
