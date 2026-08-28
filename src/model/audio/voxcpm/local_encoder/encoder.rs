//! `LocalEncoder` — VoxCPM2's `feat_encoder` (local encoder / "locenc"),
//! inference only.
//!
//! ```text
//! x [B, T, 4, 64]
//!   -> in_proj: Linear(64 -> 1024, WITH bias)          [B, T, 4, 1024]
//!   -> prepend special_token, broadcast [1,1,1,1024]   [B, T, 5, 1024]
//!      -> [B, T, 5, 1024] = [CLS, p0, p1, p2, p3] per (batch, frame)
//!   -> reshape [(B*T), 5, 1024]      each (batch, frame) is independent
//!   -> 12x pre-norm transformer layer (bidirectional GQA 16/2, head_dim 128)
//!   -> final RmsNorm over all 5 positions
//!   -> CLS-pool: take position 0, reshape            [B, T, 1024]
//! ```
//!
//! Patch folding (`[T, P, D]` -> `[T, 4, 64]`) happens upstream in the
//! audio-feature stage; this module takes the input already folded and does
//! NOT implement that fold.
//!
//! Built from plain [`Var<R>`]-wrapped weights (`requires_grad = false`)
//! rather than autograd-tracked training params — same inference-only
//! posture as `AudioVaeEncoder`/`AudioVaeDecoder` in this module.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer;
use crate::model::traits::ModelClient;
use crate::nn::{
    LoraTargets, MaybeLoraLinear, Module, RmsNorm, RoPE, adapt_if_targeted, child_params,
    extend_named, load_lora_child, push_projection_name, var_contiguous,
};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

pub struct LocalEncoder<R: Runtime> {
    /// [`MaybeLoraLinear`], not plain `Linear`: a GGUF stores this
    /// projection block-quantized, and the quantized variant multiplies it
    /// PACKED through `quant_matmul` rather than expanding it to dense F32 at
    /// load. Its 4-D `[B, T, num_patches, patch_dim]` input is fine —
    /// `quant_matmul`'s contract is `[..., M, K]`, the same leading-dims rule
    /// dense `matmul` follows, so nothing reshapes here. `MaybeLoraLinear`
    /// additionally lets this projection carry a LoRA adapter.
    pub(crate) in_proj: MaybeLoraLinear<R>,
    /// `[1, 1, 1, hidden_dim]`, broadcast to `[B, T, 1, hidden_dim]` and
    /// prepended along the patch axis.
    ///
    /// DENSE, deliberately: it is a learned constant that is concatenated,
    /// never multiplied, so there is no packed kernel it could feed.
    pub(crate) special_token: Var<R>,
    pub(crate) layers: Vec<BidirectionalLayer<R>>,
    pub(crate) norm: RmsNorm<R>,
    pub(crate) rope: RoPE<R>,
    pub(crate) hidden_dim: usize,
    /// Run every layer through
    /// [`BidirectionalLayer::forward_checkpointed`] instead of
    /// [`BidirectionalLayer::forward`]. `false` by default, so inference
    /// pays nothing — see [`Self::set_activation_checkpointing`].
    pub(crate) activation_checkpointing: bool,
}

impl<R: Runtime<DType = DType>> LocalEncoder<R> {
    /// Turn activation checkpointing on or off for every layer in this stack.
    ///
    /// `on` trades ~33% extra compute for dropping each layer's
    /// intermediates during the forward pass and recomputing them during
    /// backward, which is what caps training VRAM. Default is `off`, so an
    /// inference path pays nothing.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        self.activation_checkpointing = on;
    }

    /// Whether this stack runs its layers with activation checkpointing.
    pub fn activation_checkpointing(&self) -> bool {
        self.activation_checkpointing
    }

    /// `x: [B, T, num_patches, patch_dim]` -> `[B, T, hidden_dim]`.
    ///
    /// When [`set_activation_checkpointing`](Self::set_activation_checkpointing)
    /// is on, every layer runs through
    /// [`BidirectionalLayer::forward_checkpointed`] — same ops, same order,
    /// same output values, at ~33% extra compute.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` is what `MaybeLoraLinear::forward` adds over a
        // dense `Linear::forward`, here for `in_proj` and for every
        // projection inside the layer stack.
        C: ModelClient<R> + TypeConversionOps<R>,
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
        let shape = x.shape().to_vec();
        if shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected 4D [B, T, num_patches, patch_dim], got {}D",
                    shape.len()
                ),
            });
        }
        let (batch, seq_t, num_patches, _patch_dim) = (shape[0], shape[1], shape[2], shape[3]);

        // in_proj: patch_dim -> hidden_dim, WITH bias — the only biased
        // Linear in this module.
        let projected = self.in_proj.forward(client, x)?;

        // Prepend the learned CLS/special token, broadcast [1,1,1,H] ->
        // [B, T, 1, H], then concat along the patch axis (dim 2).
        // `broadcast_to` yields a strided view; `cat` copies from its inputs
        // and refuses one, so materialize it here.
        let special = var_contiguous(
            &var_broadcast_to(&self.special_token, &[batch, seq_t, 1, self.hidden_dim])
                .map_err(Error::Numr)?,
        )?;
        let with_cls = var_cat(&[&special, &projected], 2, client).map_err(Error::Numr)?;

        // [B, T, num_patches+1, H] -> [(B*T), num_patches+1, H]: each
        // (batch, frame) pair is an independent length-5 sequence.
        let seq_len = num_patches + 1;
        let flat = var_reshape(&with_cls, &[batch * seq_t, seq_len, self.hidden_dim])
            .map_err(Error::Numr)?;

        let mut h = flat;
        for layer in &self.layers {
            h = if self.activation_checkpointing {
                layer.forward_checkpointed(&h, &self.rope)?
            } else {
                layer.forward(client, &h, &self.rope)?
            };
        }
        let h = self.norm.forward(client, &h)?;

        // CLS-pool: position 0 only, reshape [(B*T), 1, H] -> [B, T, H].
        // `narrow` yields a view over a strided slice; `reshape` needs it
        // materialized before it can reinterpret the layout.
        let cls = var_contiguous(&var_narrow(&h, 1, 0, 1).map_err(Error::Numr)?)?;
        var_reshape(&cls, &[batch, seq_t, self.hidden_dim]).map_err(Error::Numr)
    }

    /// Wrap `in_proj` and every layer's targeted projections with a fresh
    /// LoRA adapter, returning the total adapted. `prefix` mirrors
    /// `Module::named_parameters` below exactly: `in_proj` is joined
    /// straight onto `prefix`, and each layer is joined at
    /// `"encoder.layers.{i}"`.
    ///
    /// This is the entry point for adapting this sub-model DIRECTLY (a
    /// caller may adapt just `feat_encoder` on its own), so it validates
    /// every target up front with [`LoraTargets::ensure_all_match`] against
    /// this tree's OWN full candidate set (`self.named_parameters()`,
    /// joined with `prefix`) before delegating to
    /// [`Self::apply_lora_unchecked`].
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let candidates = self.lora_projection_names(prefix);
        targets.ensure_all_match(&candidates)?;
        self.apply_lora_unchecked(targets, rank, alpha, device, prefix)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — `in_proj` plus every layer's projections —
    /// INDEPENDENT of whether `in_proj` or any layer projection is dense,
    /// block-quantized, or decomposed-quantized. This is what fixes the
    /// QLoRA validation bug: on a GGUF checkpoint `in_proj` and every layer
    /// projection are block-quantized, so `named_parameters()` returns
    /// EMPTY for all of them and a valid target would be rejected as
    /// matching nothing. `special_token`/`norm` carry no
    /// [`MaybeLoraLinear`] projections, so neither contributes a name.
    /// Matches [`Self::apply_lora_unchecked`]'s walk exactly: `in_proj` is
    /// joined straight onto `prefix` via the SAME
    /// [`crate::nn::push_projection_name`] helper `apply_lora`'s
    /// [`adapt_if_targeted`] call uses, and each layer is joined at the
    /// SAME `"encoder.layers.{i}"` prefix passed to
    /// [`crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer::apply_lora`],
    /// so a path here is never built by separately hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        push_projection_name(&mut names, prefix, "in_proj");
        for (i, layer) in self.layers.iter().enumerate() {
            names.extend(
                layer.lora_projection_names(&LoraTargets::join(
                    prefix,
                    &format!("encoder.layers.{i}"),
                )),
            );
        }
        names
    }

    /// Same walk as [`Self::apply_lora`] but skips
    /// [`LoraTargets::ensure_all_match`]. Exists for a parent
    /// (`VoxCpm2Model`) that has already validated `targets` against the
    /// WHOLE model: re-validating here against only this subtree would
    /// reject a target that lives in a sibling (`base_lm`, `residual_lm`,
    /// `feat_decoder`, `aux`), even though it is perfectly valid at root.
    pub(crate) fn apply_lora_unchecked(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = adapt_if_targeted(
            &mut self.in_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "in_proj",
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            adapted += layer.apply_lora(
                targets,
                rank,
                alpha,
                device,
                &LoraTargets::join(prefix, &format!("encoder.layers.{i}")),
            )?;
        }
        Ok(adapted)
    }

    /// Write back updated adapter values for `in_proj` and every layer from
    /// an optimizer's `params` map, keeping every adapter's [`TensorId`]s.
    /// See [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix or target validation needed here
    /// — unlike [`Self::apply_lora`], lookup is by ID, not by dotted path.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.in_proj, params, "in_proj")?;
        for layer in self.layers.iter_mut() {
            written += layer.load_lora_parameters(params)?;
        }
        Ok(written)
    }
}

/// Names mirror `feat_encoder.{in_proj,special_token,encoder.layers.{i},
/// encoder.norm}` (checkpoint prefix `feat_encoder` added by
/// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)).
/// `layers`/`norm` are hardcoded under an `encoder.` segment rather than
/// their bare field names: the checkpoint nests the transformer stack under
/// `feat_encoder.encoder.*` (see
/// [`crate::model::audio::voxcpm::local_encoder::loader`]), a level this
/// struct's field names do not carry.
impl<R: Runtime<DType = DType>> Module<R> for LocalEncoder<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.in_proj);
        params.push(&self.special_token);
        for layer in &self.layers {
            params.extend(child_params(layer));
        }
        params.extend(child_params(&self.norm));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "in_proj", self.in_proj.named_parameters());
        params.push(("special_token".to_string(), &self.special_token));
        for (i, layer) in self.layers.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("encoder.layers.{i}"),
                layer.named_parameters(),
            );
        }
        extend_named(&mut params, "encoder.norm", self.norm.named_parameters());
        params
    }
}
