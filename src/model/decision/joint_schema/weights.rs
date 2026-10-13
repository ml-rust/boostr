//! `Weights`: the named tensors a joint-schema head is built from, taken
//! one by one with their shapes checked and cast to the compute dtype, and
//! assembled into the head's building blocks.

use super::blocks::{Affine, Attention, FeedForward, Norm};
use super::config::JointSchemaConfig;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

pub(super) fn model_error(reason: String) -> Error {
    Error::ModelError { reason }
}

/// The named tensors a head is built from, taken one by one with their
/// shapes checked and cast to the compute dtype.
pub(super) struct Weights<'a, R: Runtime, C> {
    client: &'a C,
    tensors: HashMap<String, Tensor<R>>,
    dtype: DType,
}

impl<'a, R: Runtime<DType = DType>, C: ModelClient<R>> Weights<'a, R, C> {
    pub(super) fn new(client: &'a C, tensors: HashMap<String, Tensor<R>>, dtype: DType) -> Self {
        Self {
            client,
            tensors,
            dtype,
        }
    }

    /// Checks that every tensor was taken.
    pub(super) fn finish(self) -> Result<()> {
        if self.tensors.is_empty() {
            return Ok(());
        }
        let mut left: Vec<&String> = self.tensors.keys().collect();
        left.sort();
        Err(model_error(format!(
            "joint schema head: unexpected tensors {left:?}"
        )))
    }

    pub(super) fn take(&mut self, key: &str, shape: &[usize]) -> Result<Tensor<R>> {
        let t = self
            .tensors
            .remove(key)
            .ok_or_else(|| model_error(format!("joint schema head: no tensor {key}")))?;
        if t.shape() != shape {
            return Err(model_error(format!(
                "joint schema head: {key} is {:?}, expected {shape:?}",
                t.shape()
            )));
        }
        if !t.dtype().is_float() {
            return Err(model_error(format!(
                "joint schema head: {key} is {:?}, expected a float dtype",
                t.dtype()
            )));
        }
        if t.dtype() == self.dtype {
            return Ok(t.contiguous()?);
        }
        Ok(self.client.cast(&t, self.dtype)?)
    }

    pub(super) fn norm(&mut self, prefix: &str, width: usize) -> Result<Norm<R>> {
        Ok(Norm {
            weight: self.take(&format!("{prefix}.weight"), &[width])?,
            bias: self.take(&format!("{prefix}.bias"), &[width])?,
        })
    }

    pub(super) fn affine(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Affine<R>> {
        let weight = self.take(&format!("{prefix}.weight"), &[output, input])?;
        let bias = if bias {
            Some(self.take(&format!("{prefix}.bias"), &[output])?)
        } else {
            None
        };
        Ok(Affine {
            weight_t: weight.transpose(0, 1)?.contiguous()?,
            bias,
        })
    }

    pub(super) fn feedforward(
        &mut self,
        inner: &str,
        outer: &str,
        c: &JointSchemaConfig,
    ) -> Result<FeedForward<R>> {
        Ok(FeedForward {
            inner: self.affine(inner, c.width, c.feedforward, true)?,
            outer: self.affine(outer, c.feedforward, c.width, true)?,
        })
    }

    /// A torch `nn.MultiheadAttention` under `prefix`: `in_proj_weight`
    /// rows `[0, W)` query, `[W, 2W)` key, `[2W, 3W)` value.
    pub(super) fn attention(
        &mut self,
        prefix: &str,
        c: &JointSchemaConfig,
    ) -> Result<Attention<R>> {
        let w = c.width;
        let in_w = self.take(&format!("{prefix}.in_proj_weight"), &[3 * w, w])?;
        let in_b = self.take(&format!("{prefix}.in_proj_bias"), &[3 * w])?;
        let part = |i: usize| -> Result<Affine<R>> {
            Ok(Affine {
                weight_t: in_w.narrow(0, i * w, w)?.transpose(0, 1)?.contiguous()?,
                bias: Some(in_b.narrow(0, i * w, w)?.contiguous()?),
            })
        };
        Ok(Attention {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: self.affine(&format!("{prefix}.out_proj"), w, w, true)?,
            heads: c.heads,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::blocks::tests::{HostWeights, values};
    use super::super::config::FIELD_TYPES;
    use super::super::head::JointSchemaHead;
    use super::super::item::{FieldKind, JointOption, JointSchemaItem};
    use super::super::load::JointScalars;
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use std::ops::Range;

    /// Tensor names, shapes, and whether the values sit near 1.
    type Names = Vec<(String, Vec<usize>, bool)>;

    pub(crate) const SCALARS: JointScalars = JointScalars {
        prior_logit_scale: 0.75,
        joint_logit_scale: 5.0,
        residual_gate: -0.5,
    };

    /// Record length of [`tiny_items`] and [`tiny_inputs`].
    pub(crate) const RECORD: usize = 12;

    pub(crate) fn tiny_config(hidden_size: usize) -> JointSchemaConfig {
        JointSchemaConfig {
            hidden_size,
            width: 8,
            routing_layers: 2,
            layers: 2,
            heads: 2,
            feedforward: 12,
        }
    }

    /// Every tensor of a head of `c`: name, shape, values. LayerNorm
    /// weights sit near 1.
    pub(crate) fn tiny_weights(c: &JointSchemaConfig) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let (h, w, f) = (c.hidden_size, c.width, c.feedforward);
        let mut names: Names = Vec::new();
        let norm = |names: &mut Names, p: String, d: usize| {
            names.push((format!("{p}.weight"), vec![d], true));
            names.push((format!("{p}.bias"), vec![d], false));
        };
        let lin = |names: &mut Names, p: String, i: usize, o: usize| {
            names.push((format!("{p}.weight"), vec![o, i], false));
            names.push((format!("{p}.bias"), vec![o], false));
        };
        let attn = |names: &mut Names, p: String| {
            names.push((format!("{p}.in_proj_weight"), vec![3 * w, w], false));
            names.push((format!("{p}.in_proj_bias"), vec![3 * w], false));
            names.push((format!("{p}.out_proj.weight"), vec![w, w], false));
            names.push((format!("{p}.out_proj.bias"), vec![w], false));
        };
        norm(&mut names, "hidden_norm".into(), h);
        for p in [
            "memory_projection",
            "question_projection",
            "option_question_projection",
            "global_projection",
            "option_context_projection",
            "option_lexical_projection",
        ] {
            names.push((format!("{p}.weight"), vec![w, h], false));
        }
        names.push(("type_embedding.weight".into(), vec![FIELD_TYPES, w], false));
        for i in 0..c.routing_layers {
            let p = format!("evidence_layers.{i}");
            norm(&mut names, format!("{p}.query_norm"), w);
            norm(&mut names, format!("{p}.memory_norm"), w);
            attn(&mut names, format!("{p}.attention"));
            norm(&mut names, format!("{p}.feedforward_norm"), w);
            lin(&mut names, format!("{p}.feedforward.0"), w, f);
            lin(&mut names, format!("{p}.feedforward.3"), f, w);
        }
        norm(&mut names, "option_summary_norm".into(), w);
        for i in 0..c.layers {
            let p = format!("layers.{i}");
            attn(&mut names, format!("{p}.self_attn"));
            attn(&mut names, format!("{p}.multihead_attn"));
            lin(&mut names, format!("{p}.linear1"), w, f);
            lin(&mut names, format!("{p}.linear2"), f, w);
            for n in ["norm1", "norm2", "norm3"] {
                norm(&mut names, format!("{p}.{n}"), w);
            }
        }
        norm(&mut names, "field_norm".into(), w);
        norm(&mut names, "option_norm".into(), w);
        lin(&mut names, "residual_scorer.0".into(), 4 * w, w);
        lin(&mut names, "residual_scorer.3".into(), w, 1);
        names
            .into_iter()
            .enumerate()
            .map(|(seed, (name, shape, near_one))| {
                let v = values(shape.iter().product(), seed as u64 + 3);
                let v = if near_one {
                    v.iter().map(|x| 1.0 + 0.4 * x).collect()
                } else {
                    v
                };
                (name, shape, v)
            })
            .collect()
    }

    /// [`tiny_weights`] as host f64 values.
    pub(crate) fn tiny_host(c: &JointSchemaConfig) -> HostWeights {
        tiny_weights(c)
            .into_iter()
            .map(|(name, _, v)| (name, v.iter().map(|&x| f64::from(x)).collect()))
            .collect()
    }

    pub(crate) fn tiny_tensors(
        device: &CpuDevice,
        c: &JointSchemaConfig,
    ) -> HashMap<String, Tensor<CpuRuntime>> {
        tiny_weights(c)
            .into_iter()
            .map(|(name, shape, v)| {
                let t = Tensor::<CpuRuntime>::from_slice(&v, &shape, device).unwrap();
                (name, t)
            })
            .collect()
    }

    /// The F32 head of [`tiny_weights`] and [`SCALARS`].
    pub(crate) fn tiny_head(
        client: &CpuClient,
        device: &CpuDevice,
        hidden_size: usize,
    ) -> JointSchemaHead<CpuRuntime> {
        let c = tiny_config(hidden_size);
        JointSchemaHead::from_tensors(client, c, tiny_tensors(device, &c), SCALARS, DType::F32)
            .unwrap()
    }

    /// Fields over a record of [`RECORD`] positions: a three-option
    /// Choice, a Noul, and a one-option Choice whose spans overlap the
    /// first field's.
    pub(crate) fn tiny_items() -> Vec<JointSchemaItem> {
        let field = |kind, question: Range<usize>, spans: &[Range<usize>]| JointSchemaItem {
            kind,
            question,
            options: spans
                .iter()
                .map(|s| JointOption {
                    span: s.clone(),
                    ids: s.clone().map(|p| (p * 3 % 11) as u32).collect(),
                })
                .collect(),
        };
        vec![
            field(FieldKind::Choice, 1..3, &[3..5, 5..6, 6..9]),
            field(FieldKind::Noul, 9..10, &[10..11, 11..12]),
            field(FieldKind::Choice, 0..1, std::slice::from_ref(&(2..4))),
        ]
    }

    /// Hidden rows `[RECORD, hidden_size]` and lexical rows for `items`.
    pub(crate) fn tiny_inputs(
        device: &CpuDevice,
        hidden_size: usize,
        items: &[JointSchemaItem],
    ) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let ids = JointSchemaHead::<CpuRuntime>::lexical_ids(items).len();
        let hidden: Vec<f32> = values(RECORD * hidden_size, 901)
            .iter()
            .map(|v| 4.0 * v)
            .collect();
        let lexical = values(ids * hidden_size, 907);
        (
            Tensor::<CpuRuntime>::from_slice(&hidden, &[RECORD, hidden_size], device).unwrap(),
            Tensor::<CpuRuntime>::from_slice(&lexical, &[ids, hidden_size], device).unwrap(),
        )
    }
}
