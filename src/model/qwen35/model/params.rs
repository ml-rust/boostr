//! Parameter access for a [`Qwen35Model`], in loader-layout names.
//!
//! Names follow `Qwen35Model::from_varbuilder`: `model.embed_tokens.weight`,
//! then per layer `model.layers.{i}.input_layernorm.weight`, the mixer
//! under `self_attn` or `linear_attn`, `post_attention_layernorm.weight`
//! and `mlp.{gate,up,down}_proj`, then `model.norm.weight` and
//! `lm_head.*`. Each mixer reports its own names: see
//! [`Qwen35AttentionBlock::named_parameters`](crate::model::hybrid::Qwen35AttentionBlock::named_parameters)
//! and [`GdnBlock::named_parameters`](crate::model::hybrid::GdnBlock::named_parameters).
//!
//! A tied head is a dense `lm_head` over the embedding table's storage. It
//! is no parameter of its own: the training forward runs it through the
//! embedding's `Var`, and the list omits `lm_head`.

use super::build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};
use crate::nn::{MaybeQuantLinear, MaybeRotatedLinear, Module, RmsNorm, RotatedMlp, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::TensorId;

impl<R: Runtime<DType = DType>> Qwen35Model<R> {
    /// Named parameters in loader-layout names, in model order. See the
    /// module docs.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "model.embed_tokens",
            self.embed_tokens.named_parameters(),
        );
        for (i, block) in self.blocks.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("model.layers.{i}"),
                block_named(block),
            );
        }
        extend_named(
            &mut params,
            "model.norm",
            Module::named_parameters(&self.norm),
        );
        if self.tied_head().is_none() {
            extend_named(&mut params, "lm_head", self.lm_head.named_parameters());
        }
        params
    }

    /// All parameters with their stable autograd ids, in the order of
    /// [`Self::named_parameters`].
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.named_parameters()
            .into_iter()
            .map(|(_, var)| (var.id(), var))
            .collect()
    }

    /// Parameters that require grad.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// The embedding's `Var` when `lm_head` is a dense head over the
    /// embedding table's storage.
    pub(super) fn tied_head(&self) -> Option<&Var<R>> {
        let MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(head)) = &self.lm_head else {
            return None;
        };
        if self.embed_tokens.is_rotated() {
            return None;
        }
        let table = self.embed_tokens.weight()?;
        let (h, t) = (head.weight().tensor(), table.tensor());
        (h.ptr() == t.ptr() && h.shape() == t.shape() && h.dtype() == t.dtype()).then_some(table)
    }
}

impl<R: Runtime<DType = DType>> Module<R> for Qwen35Model<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Qwen35Model::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        Qwen35Model::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Qwen35Model::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        Qwen35Model::trainable_parameters(self)
    }
}

/// One layer's parameters, named relative to `model.layers.{i}`.
pub(super) fn block_named<R: Runtime<DType = DType>>(
    block: &Qwen35Block<R>,
) -> Vec<(String, &Var<R>)> {
    match block {
        Qwen35Block::Attention(layer) => attention_layer_named(layer),
        Qwen35Block::Gdn(layer) => gdn_layer_named(layer),
    }
}

pub(super) fn attention_layer_named<R: Runtime<DType = DType>>(
    layer: &Qwen35AttentionLayer<R>,
) -> Vec<(String, &Var<R>)> {
    layer_named(
        &layer.attn_norm,
        ("self_attn", layer.mixer.named_parameters()),
        &layer.post_attention_norm,
        &layer.mlp,
    )
}

pub(super) fn gdn_layer_named<R: Runtime<DType = DType>>(
    layer: &Qwen35GdnLayer<R>,
) -> Vec<(String, &Var<R>)> {
    layer_named(
        &layer.attn_norm,
        ("linear_attn", layer.mixer.named_parameters()),
        &layer.post_attention_norm,
        &layer.mlp,
    )
}

fn layer_named<'a, R: Runtime<DType = DType>>(
    attn_norm: &'a RmsNorm<R>,
    mixer: (&str, Vec<(String, &'a Var<R>)>),
    post_attention_norm: &'a RmsNorm<R>,
    mlp: &'a RotatedMlp<R>,
) -> Vec<(String, &'a Var<R>)> {
    let mut params = Vec::new();
    extend_named(
        &mut params,
        "input_layernorm",
        Module::named_parameters(attn_norm),
    );
    extend_named(&mut params, mixer.0, mixer.1);
    extend_named(
        &mut params,
        "post_attention_layernorm",
        Module::named_parameters(post_attention_norm),
    );
    extend_named(&mut params, "mlp.gate_proj", mlp.gate().named_parameters());
    extend_named(&mut params, "mlp.up_proj", mlp.up().named_parameters());
    extend_named(&mut params, "mlp.down_proj", mlp.down().named_parameters());
    params
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{Head, chunked_model, tiny_model};
    use super::*;
    use crate::test_utils::cpu_setup;

    /// Every weight of the tiny model under its loader name, ids aligned
    /// with `parameters`, nothing trainable.
    #[test]
    fn names_follow_the_loader_layout() {
        let (_, device) = cpu_setup();
        let model = tiny_model(&device, 0x3a7e_0100);
        let named = model.named_parameters();
        let names: Vec<&str> = named.iter().map(|(name, _)| name.as_str()).collect();
        let attn = (0..model.blocks.len())
            .find(|&i| model.blocks[i].is_attention())
            .unwrap();
        let gdn = 1 - attn;
        for want in [
            "model.embed_tokens.weight".to_string(),
            format!("model.layers.{attn}.input_layernorm.weight"),
            format!("model.layers.{attn}.self_attn.q_proj.weight"),
            format!("model.layers.{attn}.self_attn.k_norm.weight"),
            format!("model.layers.{gdn}.linear_attn.in_proj_qkv.weight"),
            format!("model.layers.{gdn}.post_attention_layernorm.weight"),
            format!("model.layers.{gdn}.mlp.down_proj.weight"),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ] {
            assert!(names.contains(&want.as_str()), "missing {want}");
        }
        let ids: Vec<TensorId> = named.iter().map(|(_, var)| var.id()).collect();
        let want: Vec<TensorId> = model.parameters().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, want);
        assert!(model.trainable_parameters().is_empty());
        assert_eq!(Module::parameters(&model).len(), ids.len());
    }

    /// A tied head over the embedding storage is no separate parameter.
    #[test]
    fn tied_head_is_not_listed() {
        let (_, device) = cpu_setup();
        let tied = chunked_model(&device, 0x3a7e_0101, 64, false, Head::Tied);
        assert!(tied.tied_head().is_some());
        assert!(
            !tied
                .named_parameters()
                .iter()
                .any(|(n, _)| n.starts_with("lm_head"))
        );
        let own = chunked_model(&device, 0x3a7e_0101, 64, false, Head::Own);
        assert!(own.tied_head().is_none());
    }
}
