use super::MiniCpm4Layer;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, checkpoint_with_client, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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
        let (h, mlp_out) = self.forward_with_pending_residual(client, x, None, rope)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Same computation as [`forward`](Self::forward), but accepts the
    /// PREVIOUS layer's deferred MLP output (`pending`) and defers this
    /// layer's OWN MLP output to the caller instead of finishing the final
    /// residual add here — one fewer kernel launch per layer boundary.
    /// Mirrors the deferred-residual pattern `LlamaBlock::forward_with_kv_cache`
    /// uses (`src/model/llama/model/blocks/block.rs`), which is `pub(super)`
    /// and so out of reach for an intra-doc link from here.
    ///
    /// Returns `(h, mlp_out)`: `h` is `(x + pending) + attn_out` (the
    /// residual after attention, with `pending` already folded in), and
    /// `mlp_out` is this layer's UNADDED MLP output — the caller passes it as
    /// `pending` to the NEXT layer, or folds it into the model's final norm
    /// after the last layer (same fold [`forward`] does for its own output).
    ///
    /// Each of the two fuse points (`pending` into `input_layernorm`, and the
    /// attention residual into `post_attention_layernorm`) is taken only when
    /// neither operand needs a gradient: `fused_add_forward` bypasses
    /// autograd, so fusing under a live graph would drop backprop to a LoRA
    /// adapter or an earlier layer.
    pub fn forward_with_pending_residual<C>(
        &self,
        client: &C,
        x: &Var<R>,
        pending: Option<&Var<R>>,
        rope: Option<&RoPE<R>>,
    ) -> Result<(Var<R>, Var<R>)>
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
        let (normed, x) = match pending {
            Some(prev_mlp) => self.input_layernorm.residual_norm(client, x, prev_mlp)?,
            None => {
                let normed = self.input_layernorm.forward(client, x)?;
                (normed, x.clone())
            }
        };

        let attn_out = self.self_attn.forward(client, &normed, rope)?;

        let (normed, h) = self
            .post_attention_layernorm
            .residual_norm(client, &x, &attn_out)?;

        let mlp_out = self.mlp.forward(client, &normed)?;
        Ok((h, mlp_out))
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
    pub fn forward_checkpointed<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: Option<&RoPE<R>>,
    ) -> Result<Var<R>>
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
        let rope = rope.map(RoPE::alias);
        // `x` first, then every trainable parameter: the backward pass
        // returns a gradient only for an id it was given as an input.
        let trainable = crate::nn::Module::trainable_parameters(self);
        let mut inputs: Vec<&Var<R>> = Vec::with_capacity(trainable.len() + 1);
        inputs.push(x);
        inputs.extend(trainable.iter().map(|(_, param)| *param));
        checkpoint_with_client(
            move |segment_inputs, client: &C| {
                let input = segment_inputs.first().ok_or_else(|| {
                    numr::error::Error::Internal(
                        "checkpointed MiniCpm4Layer segment received no input".to_string(),
                    )
                })?;
                layer
                    .forward::<C>(client, input, rope.as_ref())
                    .map_err(|e| {
                        numr::error::Error::Backend(format!(
                            "checkpointed MiniCpm4Layer forward: {e}"
                        ))
                    })
            },
            &inputs,
            client,
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
        kv_start: Option<&Tensor<R>>,
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
        let (h, mlp_out) = self.forward_cached_with_pending_residual(
            client, x, None, rope, kv_cache, position, kv_start,
        )?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Cross-layer counterpart to [`forward_with_pending_residual`](Self::forward_with_pending_residual),
    /// for the KV-cached attention path: same `pending`-in / `(h, mlp_out)`-out
    /// contract, only attention reads and extends `kv_cache` instead of
    /// recomputing the whole prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached_with_pending_residual<C>(
        &self,
        client: &C,
        x: &Var<R>,
        pending: Option<&Var<R>>,
        rope: Option<&RoPE<R>>,
        kv_cache: &mut KvCache<R>,
        position: usize,
        kv_start: Option<&Tensor<R>>,
    ) -> Result<(Var<R>, Var<R>)>
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
        let (normed, x) = match pending {
            Some(prev_mlp) => self.input_layernorm.residual_norm(client, x, prev_mlp)?,
            None => {
                let normed = self.input_layernorm.forward(client, x)?;
                (normed, x.clone())
            }
        };
        let attn_out = self
            .self_attn
            .forward_cached(client, &normed, rope, kv_cache, position, kv_start)?;

        let (normed, h) = self
            .post_attention_layernorm
            .residual_norm(client, &x, &attn_out)?;

        let mlp_out = self.mlp.forward(client, &normed)?;
        Ok((h, mlp_out))
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

#[cfg(test)]
mod alias_tests {
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::model::tests::{HIDDEN, filled, tiny_model};
    use crate::nn::{LoraTargets, Module};
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::TensorId;
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

    /// The fused residual-add + RMSNorm path (`x.requires_grad() == false`)
    /// must compute the SAME numbers as the unfused path (`true`), across the
    /// WHOLE 2-layer stack `tiny_model` builds — not just one layer.
    /// `x.requires_grad() == true` on the model input propagates through
    /// every deferred-residual add (`var_add` sets `requires_grad` when
    /// either operand does), forcing EVERY layer's `forward_with_pending_residual`
    /// down the unfused branch, and the model's own final fold too — this is
    /// what makes a single top-level flag exercise the cross-layer fusion in
    /// `MiniCpm4Model::forward`'s layer loop, not just one layer's.
    #[test]
    fn fused_add_forward_matches_unfused_residual_path() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let x_data = filled(&[1, 4, HIDDEN], 99, &device);

        let x_fused = Var::new(x_data.clone(), false);
        let out_fused = model.forward(&client, &x_fused).expect("fused forward");

        // `requires_grad(true)` on a leaf with no `grad_fn` forces the
        // unfused branch without changing a single computed value.
        let x_unfused = Var::new(x_data, true);
        let out_unfused = model.forward(&client, &x_unfused).expect("unfused forward");

        let fused = values(out_fused.tensor());
        let unfused = values(out_unfused.tensor());
        assert_eq!(fused.len(), unfused.len());
        for (a, b) in fused.iter().zip(&unfused) {
            assert!(
                (a - b).abs() < 1e-5,
                "fused vs unfused residual+norm diverged across the 2-layer stack: {a} vs {b}"
            );
        }
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
            .forward_checkpointed(&client, &x, rope.as_ref())
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
