//! Single pre-norm transformer layer for VoxCPM2's shared bidirectional
//! MiniCPM4 block stack, used by both `feat_encoder` (`local_encoder`) and
//! the local DiT (`feat_decoder`).
//!
//! Same pre-norm-attention-residual, pre-norm-MLP-residual sequence as
//! `LlamaBlock` (`RmsNorm` -> attn -> add, `RmsNorm` -> MLP -> add), with the
//! two differences this checkpoint requires: attention is
//! [`BidirectionalAttention`] (bidirectional GQA, not `LlamaAttention`'s
//! always-causal path) and residuals are plain adds — `use_mup` is `false`
//! on this checkpoint, so no muP `scale_depth/sqrt(num_layers)` factor is
//! applied (unlike some MiniCPM-lineage ports that assume it is).

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
use crate::model::traits::ModelClient;
use crate::nn::{LoraTargets, Module, RmsNorm, RoPE, child_params, extend_named};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, checkpoint_with_client, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

pub struct BidirectionalLayer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: BidirectionalAttention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: BidirectionalMlp<R>,
}

impl<R: Runtime<DType = DType>> BidirectionalLayer<R> {
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` comes from the `MaybeLoraLinear` projections
        // inside the attention and MLP sub-blocks; the norms below need
        // nothing extra.
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
    /// `numr::autograd::checkpoint_with_client` runs the segment on the
    /// caller's `client`, on both the forward pass and the recompute, so the
    /// closure needs the same bounds on `C` that [`forward`](Self::forward)
    /// needs and nothing extra on `R::Client`. Same ops in the same order as
    /// [`forward`](Self::forward), so the output values match exactly.
    ///
    /// The stored closure is `Send + Sync + 'static`, so it cannot borrow
    /// `&self`, and `C` must be `'static`. This captures [`Self::alias`] and
    /// an aliased `rope` instead, which preserves every `TensorId` — a
    /// `Clone` would mint fresh ids and orphan the adapters' gradients.
    ///
    /// Every trainable parameter this layer owns is passed alongside `x`.
    /// The recompute differentiates only with respect to the ids it was
    /// handed, so a parameter left out of that list gets NO gradient — the
    /// adapters would silently stop training. numr rejects that case at
    /// forward time.
    pub fn forward_checkpointed<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + 'static,
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
        let layer = self.alias();
        let rope = rope.alias();
        // `x` first, then every trainable parameter: the backward pass
        // returns a gradient only for an id it was given as an input.
        let trainable = Module::trainable_parameters(self);
        let mut inputs: Vec<&Var<R>> = Vec::with_capacity(trainable.len() + 1);
        inputs.push(x);
        inputs.extend(trainable.iter().map(|(_, param)| *param));
        checkpoint_with_client(
            move |segment_inputs, client: &C| {
                let input = segment_inputs.first().ok_or_else(|| {
                    numr::error::Error::Internal(
                        "checkpointed BidirectionalLayer segment received no input".to_string(),
                    )
                })?;
                layer.forward::<C>(client, input, &rope).map_err(|e| {
                    numr::error::Error::Backend(format!(
                        "checkpointed BidirectionalLayer forward: {e}"
                    ))
                })
            },
            &inputs,
            client,
        )
        .map_err(Error::Numr)
    }

    /// Delegate to `BidirectionalAttention::apply_lora` and
    /// `BidirectionalMlp::apply_lora`, summing their counts. `prefix` is the
    /// dotted path the owning `LocalEncoder`/`LocalDit` would pass to
    /// `extend_named` for this layer, extended here by `"self_attn"`/`"mlp"`
    /// exactly as `Module::named_parameters` extends it above. No
    /// zero-match check here: see `BidirectionalAttention::apply_lora`'s
    /// doc comment.
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
    /// `prefix`, delegating to `BidirectionalAttention::lora_projection_names`
    /// and `BidirectionalMlp::lora_projection_names` at the SAME
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

    /// Delegate to `BidirectionalAttention::load_lora_parameters` and
    /// `BidirectionalMlp::load_lora_parameters`, summing their counts. No
    /// prefix needed — unlike [`Self::apply_lora`], lookup is by ID, not by
    /// dotted path.
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
/// `post_attention_layernorm`, `mlp.*`) — this matches the shared
/// `{layer_prefix}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::bidirectional::loader`]) exactly, so the
/// owning `LocalEncoder`/`LocalDit` need only prefix by `{layer_prefix}` (a
/// numeric layer index under `encoder.layers`/`decoder.layers`) to reach the
/// full checkpoint key.
impl<R: Runtime<DType = DType>> Module<R> for BidirectionalLayer<R> {
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
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{HEAD_DIM, HIDDEN_DIM, layer, t};
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};
    use numr::runtime::cpu::CpuRuntime;
    use std::collections::HashMap;

    const SEQ: usize = 4;

    fn values(tensor: &Tensor<CpuRuntime>) -> Vec<f32> {
        tensor.contiguous().expect("contiguous").to_vec::<f32>()
    }

    /// The load-bearing test for activation checkpointing.
    ///
    /// A forward-only comparison would pass on a `forward_checkpointed` that
    /// never reconstructs a usable graph, which is the whole failure mode.
    /// The gradient half is what proves the recompute rebuilt the segment:
    /// every LoRA adapter and the input must receive the SAME gradient on
    /// both paths.
    ///
    /// Equality is EXACT, not approximate. `checkpoint` re-runs the same ops
    /// in the same order on the same values, and the loss is a plain sum, so
    /// the incoming `grad_output` is all ones — the extra `mul` the
    /// checkpoint backward inserts multiplies by exactly 1.0 and rounds
    /// nothing. Any drift here means the recomputed graph is not the graph
    /// the forward pass ran.
    #[test]
    fn checkpointed_forward_matches_forward_in_values_and_lora_gradients() {
        let (client, device) = cpu_setup();
        let mut layer = layer(1.0, &device);
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

        // `LoraLinear::new` zeroes `lora_b`, which would make d(loss)/d(lora_a)
        // exactly zero and that half of the comparison vacuous. Overwrite both
        // adapters with non-degenerate values; `load_lora_parameters` keeps
        // every `TensorId`, so the ids collected below stay valid.
        let seeded: HashMap<TensorId, Tensor<CpuRuntime>> = Module::trainable_parameters(&layer)
            .into_iter()
            .enumerate()
            .map(|(i, (id, var))| (id, t(var.shape(), 0.6 + i as f32, &device)))
            .collect();
        let written = layer
            .load_lora_parameters(&seeded)
            .expect("load_lora_parameters must write every seeded adapter");
        assert_eq!(
            written,
            seeded.len(),
            "every seeded adapter must be written"
        );

        let rope = RoPE::<CpuRuntime>::precompute_freqs(32, HEAD_DIM, 10000.0, None, &device)
            .expect("rope")
            .narrow_positions(SEQ)
            .expect("narrow");
        let x = Var::new(t(&[1, SEQ, HIDDEN_DIM], 0.3, &device), true);

        let out_plain = layer.forward(&client, &x, &rope).expect("forward");
        let loss_plain = var_sum(&out_plain, &[], false, &client).expect("sum");
        let grads_plain = backward(&loss_plain, &client).expect("backward");

        let out_ckpt = layer
            .forward_checkpointed(&client, &x, &rope)
            .expect("forward_checkpointed");
        let loss_ckpt = var_sum(&out_ckpt, &[], false, &client).expect("sum");
        let grads_ckpt = backward(&loss_ckpt, &client).expect("checkpointed backward");

        assert_eq!(
            values(out_plain.tensor()),
            values(out_ckpt.tensor()),
            "forward_checkpointed must produce the same values as forward"
        );

        let adapters = Module::trainable_parameters(&layer);
        assert_eq!(
            adapters.len(),
            4,
            "two adapted projections, lora_a + lora_b"
        );
        let mut any_nonzero = false;
        for (id, var) in &adapters {
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
                plain.len(),
                var.tensor().numel(),
                "adapter gradient must match its parameter's element count"
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
