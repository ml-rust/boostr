//! The weight edits `config.block` asks for: the Gemma norm offset, the
//! query-scale fold, and the partial/interleaved RoPE row reorder.

use super::fused::take_dense;
use crate::error::{Error, Result};
use crate::model::config::{BlockConfig, UniversalConfig};
use crate::nn::VarMap;
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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
pub(super) fn apply_block_edits<R: Runtime<DType = DType>>(
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

#[cfg(test)]
mod tests {
    use super::super::restructure_hf_checkpoint;
    use super::super::test_support::{config, rows, tagged};
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

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
