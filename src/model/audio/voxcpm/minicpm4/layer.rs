//! Single pre-norm decoder layer for VoxCPM2's MiniCPM4.
//!
//! `RmsNorm` -> causal attention -> add, `RmsNorm` -> SwiGLU MLP -> add.
//!
//! Residuals are PLAIN adds: `use_mup` is `false` on this checkpoint, so no
//! muP `scale_depth/sqrt(num_layers)` factor is applied (unlike some
//! MiniCPM-lineage ports that assume it always is). `scale_emb` (12.0) is
//! inactive for the same reason — the reference applies it only under muP —
//! and neither knob has an inert branch here;
//! [`MiniCpm4Config`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config)
//! rejects a `use_mup=true` checkpoint outright rather than letting this
//! layer compute a different model in silence.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
use crate::model::traits::ModelClient;
use crate::nn::{LoraTargets, Module, RmsNorm, RoPE, child_params, extend_named};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, checkpoint, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

pub struct MiniCpm4Layer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: MiniCpm4Attention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: MiniCpm4Mlp<R>,
}

impl<R: Runtime<DType = DType>> MiniCpm4Layer<R> {
    /// `x: [batch, seq, hidden]` -> `[batch, seq, hidden]`.
    ///
    /// `rope` is `None` only for a NoPE (`no_rope`) stack, which has no table;
    /// [`MiniCpm4Attention`] rejects a `None` it is not entitled to.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: Option<&RoPE<R>>) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Same result as [`forward`](Self::forward), computed with activation
    /// checkpointing: the layer's intermediates are dropped during the
    /// forward pass and recomputed during backward.
    ///
    /// Costs ~33% extra compute. Call it only on a training pass; inference
    /// must use [`forward`](Self::forward) and pay nothing.
    ///
    /// Takes no `client`: `numr::autograd::checkpoint` runs the segment with
    /// `R::default_client` for the input's device, on both the forward pass
    /// and the recompute. Same ops in the same order as
    /// [`forward`](Self::forward), so the output values match exactly.
    ///
    /// The closure `checkpoint` stores is `Send + Sync + 'static`, so it
    /// cannot borrow `&self`. This captures [`Self::alias`] and an aliased
    /// `rope` instead, which preserves every `TensorId` — a `Clone` would
    /// mint fresh ids and orphan the adapters' gradients.
    ///
    /// Every trainable parameter this layer owns is passed to `checkpoint`
    /// alongside `x`. `checkpoint`'s backward prunes its re-entrant pass to
    /// exactly the ids it was handed, so a parameter left out of that list
    /// gets NO gradient — the adapters would silently stop training.
    pub fn forward_checkpointed(&self, x: &Var<R>, rope: Option<&RoPE<R>>) -> Result<Var<R>>
    where
        R::Client: ModelClient<R>
            + TypeConversionOps<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let layer = self.alias();
        let rope = rope.map(RoPE::alias);
        // `x` first, then every trainable parameter: `checkpoint`'s backward
        // returns a gradient only for an id it was given as an input.
        let trainable = Module::trainable_parameters(self);
        let mut inputs: Vec<&Var<R>> = Vec::with_capacity(trainable.len() + 1);
        inputs.push(x);
        inputs.extend(trainable.iter().map(|(_, param)| *param));
        checkpoint(
            move |segment_inputs, client: &R::Client| {
                let input = segment_inputs.first().ok_or_else(|| {
                    numr::error::Error::Internal(
                        "checkpointed MiniCpm4Layer segment received no input".to_string(),
                    )
                })?;
                layer
                    .forward::<R::Client>(client, input, rope.as_ref())
                    .map_err(|e| {
                        numr::error::Error::Backend(format!(
                            "checkpointed MiniCpm4Layer forward: {e}"
                        ))
                    })
            },
            &inputs,
        )
        .map_err(Error::Numr)
    }

    /// KV-cached variant of [`forward`](Self::forward): `x: [batch, seq,
    /// hidden]` covering absolute positions `position..position + seq` ->
    /// `[batch, seq, hidden]`.
    ///
    /// Identical residual/norm/MLP structure — only attention differs, reading
    /// and extending `kv_cache` instead of recomputing the whole prefix.
    pub fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: Option<&RoPE<R>>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self
            .self_attn
            .forward_cached(client, &normed, rope, kv_cache, position)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Delegate to [`MiniCpm4Attention::apply_lora`](super::attention::MiniCpm4Attention::apply_lora)
    /// and [`MiniCpm4Mlp::apply_lora`](super::mlp::MiniCpm4Mlp::apply_lora),
    /// summing their counts. `prefix` is the dotted path the owning
    /// [`super::model::MiniCpm4Model`] would pass to `extend_named` for this
    /// layer — `"layers.{i}"` — extended here by `"self_attn"`/`"mlp"`
    /// exactly as `Module::named_parameters` extends it above. No zero-match
    /// check here: see `MiniCpm4Attention::apply_lora`'s doc comment.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = self.self_attn.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "self_attn"),
        )?;
        adapted += self.mlp.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "mlp"),
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix`, delegating to `MiniCpm4Attention::lora_projection_names`
    /// and `MiniCpm4Mlp::lora_projection_names` at the SAME
    /// `"self_attn"`/`"mlp"`-joined prefixes [`Self::apply_lora`] passes
    /// them, so a path here is never built by separately hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = self
            .self_attn
            .lora_projection_names(&LoraTargets::join(prefix, "self_attn"));
        names.extend(
            self.mlp
                .lora_projection_names(&LoraTargets::join(prefix, "mlp")),
        );
        names
    }

    /// Delegate to `MiniCpm4Attention::load_lora_parameters` and
    /// `MiniCpm4Mlp::load_lora_parameters`, summing their counts. No prefix
    /// needed — unlike [`Self::apply_lora`], lookup is by ID, not by dotted
    /// path.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = self.self_attn.load_lora_parameters(params)?;
        written += self.mlp.load_lora_parameters(params)?;
        Ok(written)
    }

    /// Cheap duplicate that preserves every child's `Var<R>` `TensorId`s,
    /// for capturing this layer by owned value in a `'static`
    /// activation-checkpointing closure — `numr::autograd::checkpoint`'s
    /// closure is `Fn(...) + Send + Sync + 'static`, so a layer cannot be
    /// borrowed into it. Every child routes through its own `alias()`,
    /// never [`Clone`], so the optimizer, keyed by `TensorId`, still sees
    /// the original parameters' gradients.
    pub fn alias(&self) -> Self {
        Self {
            input_layernorm: self.input_layernorm.alias(),
            self_attn: self.self_attn.alias(),
            post_attention_layernorm: self.post_attention_layernorm.alias(),
            mlp: self.mlp.alias(),
        }
    }
}

/// Names ARE the field names (`input_layernorm`, `self_attn.*`,
/// `post_attention_layernorm`, `mlp.*`) — this matches the
/// `{prefix}.layers.{i}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::minicpm4::loader`]) exactly, so the
/// owning [`MiniCpm4Model`](super::model::MiniCpm4Model) need only prefix
/// by `layers.{i}` to reach the full checkpoint key.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Layer<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.input_layernorm);
        params.extend(child_params(&self.self_attn));
        params.extend(child_params(&self.post_attention_layernorm));
        params.extend(child_params(&self.mlp));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "input_layernorm",
            self.input_layernorm.named_parameters(),
        );
        extend_named(&mut params, "self_attn", self.self_attn.named_parameters());
        extend_named(
            &mut params,
            "post_attention_layernorm",
            self.post_attention_layernorm.named_parameters(),
        );
        extend_named(&mut params, "mlp", self.mlp.named_parameters());
        params
    }
}

#[cfg(test)]
mod alias_tests {
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::model::tests::{HIDDEN, filled, tiny_model};
    use crate::nn::LoraTargets;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};
    use numr::runtime::cpu::CpuRuntime;
    use std::collections::HashMap;

    /// The whole point of a layer's `alias()`: prove it preserves every
    /// `Var<R>` `TensorId`, including a LoRA adapter's, not mints fresh
    /// ones like `Clone` would. If a future edit swaps `.alias()` for
    /// `.clone()` anywhere along the chain this test exercises
    /// (`MiniCpm4Layer` -> `MiniCpm4Attention` -> `MaybeLoraLinear` ->
    /// `LoraLinear`), this test must fail — a fresh id would silently
    /// orphan the adapter's gradient from the optimizer's `TensorId`-keyed
    /// state, which is exactly the trap `numr::autograd::Var::alias`'s doc
    /// comment warns about.
    #[test]
    fn alias_preserves_lora_adapter_ids_through_a_full_layer() {
        let device = <CpuRuntime as Runtime>::default_device();
        let mut model = tiny_model(&device);
        let layer = &mut model.layers[0];
        layer
            .self_attn
            .apply_lora(&LoraTargets::new(["q_proj"]), 2, 4.0, &device, "self_attn")
            .expect("apply_lora on q_proj must succeed");

        let aliased = layer.alias();

        let (orig_a, orig_b) = layer
            .self_attn
            .q_proj
            .adapters()
            .expect("q_proj carries a LoRA adapter after apply_lora");
        let (alias_a, alias_b) = aliased
            .self_attn
            .q_proj
            .adapters()
            .expect("aliased q_proj must still carry the adapter");
        assert_eq!(orig_a.id(), alias_a.id(), "lora_a id must survive alias()");
        assert_eq!(orig_b.id(), alias_b.id(), "lora_b id must survive alias()");

        // Every other Var-bearing child must alias too, not just the adapter.
        assert_eq!(
            layer.input_layernorm.weight().id(),
            aliased.input_layernorm.weight().id(),
            "input_layernorm weight id must survive alias()"
        );
    }

    fn values(tensor: &Tensor<CpuRuntime>) -> Vec<f32> {
        tensor.contiguous().expect("contiguous").to_vec::<f32>()
    }

    /// [`MiniCpm4Layer::forward_checkpointed`] must be
    /// [`MiniCpm4Layer::forward`] in values AND in gradients.
    ///
    /// A forward-only comparison would pass on a `forward_checkpointed` that
    /// never reconstructs a usable graph, which is the whole failure mode.
    /// Equality is EXACT: `checkpoint` re-runs the same ops in the same order
    /// on the same values, and the loss is a plain sum, so the incoming
    /// `grad_output` is all ones and the extra `mul` the checkpoint backward
    /// inserts multiplies by exactly 1.0.
    #[test]
    fn checkpointed_forward_matches_forward_in_values_and_lora_gradients() {
        let (client, device) = cpu_setup();
        let mut model = tiny_model(&device);
        let rope = model.rope.as_ref().map(RoPE::alias);
        let layer = &mut model.layers[0];
        let adapted = layer
            .apply_lora(
                &LoraTargets::new(["q_proj", "down_proj"]),
                2,
                4.0,
                &device,
                "",
            )
            .expect("apply_lora must adapt q_proj and down_proj");
        assert_eq!(adapted, 2, "expected one attention and one MLP projection");

        // `LoraLinear::new` zeroes `lora_b`, which would leave
        // d(loss)/d(lora_a) exactly zero and that half of the comparison
        // vacuous. Overwrite both adapters with non-degenerate values;
        // `load_lora_parameters` keeps every `TensorId`.
        let seeded: HashMap<TensorId, Tensor<CpuRuntime>> = Module::trainable_parameters(layer)
            .into_iter()
            .enumerate()
            .map(|(i, (id, var))| (id, filled(var.shape(), 7 + i, &device)))
            .collect();
        let written = layer
            .load_lora_parameters(&seeded)
            .expect("load_lora_parameters must write every seeded adapter");
        assert_eq!(
            written,
            seeded.len(),
            "every seeded adapter must be written"
        );

        let x = Var::new(filled(&[1, 4, HIDDEN], 99, &device), true);

        let out_plain = layer.forward(&client, &x, rope.as_ref()).expect("forward");
        let loss_plain = var_sum(&out_plain, &[], false, &client).expect("sum");
        let grads_plain = backward(&loss_plain, &client).expect("backward");

        let out_ckpt = layer
            .forward_checkpointed(&x, rope.as_ref())
            .expect("forward_checkpointed");
        let loss_ckpt = var_sum(&out_ckpt, &[], false, &client).expect("sum");
        let grads_ckpt = backward(&loss_ckpt, &client).expect("checkpointed backward");

        assert_eq!(
            values(out_plain.tensor()),
            values(out_ckpt.tensor()),
            "forward_checkpointed must produce the same values as forward"
        );

        let adapters = Module::trainable_parameters(layer);
        assert_eq!(
            adapters.len(),
            4,
            "two adapted projections, lora_a + lora_b"
        );
        let mut any_nonzero = false;
        for (id, _) in &adapters {
            let plain = values(
                grads_plain
                    .get(*id)
                    .expect("forward must produce an adapter gradient"),
            );
            let ckpt = values(
                grads_ckpt
                    .get(*id)
                    .expect("forward_checkpointed must produce an adapter gradient"),
            );
            assert_eq!(
                plain, ckpt,
                "adapter gradient must match between forward and forward_checkpointed"
            );
            any_nonzero |= plain.iter().any(|g| *g != 0.0);
        }
        assert!(
            any_nonzero,
            "an all-zero adapter gradient would pass this test vacuously"
        );

        let gx_plain = values(grads_plain.get(x.id()).expect("input gradient"));
        let gx_ckpt = values(
            grads_ckpt
                .get(x.id())
                .expect("checkpointed input gradient — the segment input id must survive"),
        );
        assert_eq!(
            gx_plain, gx_ckpt,
            "input gradient must match between forward and forward_checkpointed"
        );
    }
}
