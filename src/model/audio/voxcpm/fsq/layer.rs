//! VoxCPM2's `fsq_layer`: a finite-scalar-quantization bottleneck between the
//! `base_lm` decoder and `feat_decoder`'s DiT, plus the `stop` classifier
//! chain that shares its input width.
//!
//! Reference: `ScalarQuantizationLayer.forward`, EVAL mode only. The
//! reference's training branch runs a straight-through estimator around
//! `torch.round`; boostr is inference-only, so that branch is dead code and
//! is deliberately NOT ported here.

use crate::error::{Error, Result};
use crate::nn::{
    LoraTargets, MaybeLoraLinear, Module, adapt_if_targeted, child_params, extend_named,
};
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_add, var_div_scalar, var_mul_scalar, var_silu, var_tanh};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

/// `fsq_layer`: `out_proj(round_ties_even(tanh(in_proj(hidden)) * scale) /
/// scale)`.
///
/// `in_proj` narrows `lm_hidden -> latent_dim` (2048 -> 512), `out_proj`
/// widens back (512 -> 2048). The intermediate `tanh` output lives in
/// `(-1, 1)`; scaling by `scale` (9) and rounding to the nearest integer
/// before dividing back snaps each of the 512 channels onto one of 19 evenly
/// spaced levels, `k / 9` for `k` in `-9..=9`.
///
/// Both projections are [`MaybeLoraLinear`], not plain `Linear`: a GGUF
/// stores them block-quantized, and the quantized variant multiplies the
/// weight PACKED through `quant_matmul` instead of expanding it to dense F32
/// at load. A safetensors checkpoint yields the `Standard` variant and runs
/// exactly the dense path it always did. `MaybeLoraLinear` additionally lets
/// either projection carry a LoRA adapter.
pub struct ScalarQuantization<R: Runtime> {
    in_proj: MaybeLoraLinear<R>,
    out_proj: MaybeLoraLinear<R>,
    /// Rounding-grid divisor (`FsqConfig::scale`, 9 on the verified
    /// checkpoint).
    scale: f32,
}

impl<R: Runtime<DType = DType>> ScalarQuantization<R> {
    /// Wrap already-loaded `in_proj`/`out_proj` weights with `scale`. Use
    /// [`crate::model::audio::voxcpm::fsq::loader`] to build one from a
    /// checkpoint.
    pub fn new(in_proj: MaybeLoraLinear<R>, out_proj: MaybeLoraLinear<R>, scale: f32) -> Self {
        Self {
            in_proj,
            out_proj,
            scale,
        }
    }

    /// `fsq_layer.forward` (eval mode). Accepts either rank-3 `[b, T,
    /// hidden]` (batched sequence input) or rank-2 `[b, hidden]` (the
    /// per-step decode path). Any other rank is `Error::InvalidArgument`.
    ///
    /// # Ties-to-even rounding
    ///
    /// `torch.round` is IEEE round-half-to-EVEN, which disagrees with Rust's
    /// `f32::round`/numr's [`numr::tensor::Tensor::round`]
    /// (round-half-away-from-zero) at every exact tie — e.g. `-0.5 -> -0.0`
    /// under ties-even vs `-1.0` under ties-away. This uses
    /// [`numr::ops::UnaryOps::round_ties_even`] (exposed here as
    /// `Tensor::round_ties_even`) to match the reference. Do NOT swap this
    /// for `round`: the two only agree away from `.5` boundaries.
    pub fn forward<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        // `QuantMatmulOps` + `BinaryOps` + `TypeConversionOps` are what
        // `MaybeLoraLinear::forward` needs over a dense `Linear::forward`:
        // the packed multiply, its bias add, and the decomposed-quant arm's
        // cast of activations to F32.
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + UnaryOps<R> + BinaryOps<R>,
    {
        match hidden.shape().len() {
            2 | 3 => {}
            rank => {
                return Err(Error::InvalidArgument {
                    arg: "hidden",
                    reason: format!(
                        "expected rank 2 or 3, got rank {rank} ({:?})",
                        hidden.shape()
                    ),
                });
            }
        }

        let projected = self.in_proj.forward(client, hidden)?;
        let squashed = var_tanh(&projected, client).map_err(Error::Numr)?;
        let scaled = var_mul_scalar(&squashed, self.scale as f64, client).map_err(Error::Numr)?;

        // Ties-to-even, NOT ties-away — see the doc comment above.
        let rounded_tensor = scaled.tensor().round_ties_even().map_err(Error::Numr)?;

        // STRAIGHT-THROUGH ESTIMATOR: forward is exactly `round(scaled)`,
        // backward is the identity.
        //
        // `round` has zero derivative almost everywhere, so wrapping the
        // rounded tensor in a fresh detached `Var` — which is what this used
        // to do — cuts the autograd graph here. That is harmless for
        // inference and was the deliberate original choice, but it silently
        // severs EVERYTHING upstream of the quantizer from any training
        // loss: measured on the CFM loss, all of `base_lm` and
        // `fsq.in_proj` received no gradient entry at all, while every
        // module downstream of this point trained normally. A LoRA adapter
        // on `base_lm` would sit at its initial value forever and the loss
        // would still fall (the downstream modules learn), so nothing would
        // look wrong.
        //
        // `scaled + (round(scaled) - scaled)_detached` is bit-identical to
        // `round(scaled)` in the forward direction — the subtraction is
        // exact in floating point because both operands come from the same
        // value — and passes the incoming gradient through unchanged. This
        // is the same estimator the reference implementation uses in its
        // training branch, which was previously omitted here as dead code.
        let residual_tensor = rounded_tensor.sub(scaled.tensor()).map_err(Error::Numr)?;
        let residual = Var::new(residual_tensor, false);
        let rounded = var_add(&scaled, &residual, client).map_err(Error::Numr)?;

        let levels = var_div_scalar(&rounded, self.scale as f64, client).map_err(Error::Numr)?;
        self.out_proj.forward(client, &levels)
    }

    /// Wrap `in_proj`/`out_proj` that `targets` names with a fresh LoRA
    /// adapter each, returning how many were adapted. `prefix` is the
    /// dotted path the owning
    /// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)
    /// would pass to `extend_named` for this bottleneck —
    /// [`crate::model::audio::voxcpm::fsq::loader::FSQ_LAYER_PREFIX`] — so
    /// each projection's path (via [`LoraTargets::join`]) matches
    /// `named_parameters()`'s path exactly. A leaf step: no zero-match check
    /// here — see
    /// [`crate::model::audio::voxcpm::minicpm4::MiniCpm4Attention::apply_lora`]'s
    /// doc comment for why.
    pub fn apply_lora(
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
        adapted += adapt_if_targeted(
            &mut self.out_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "out_proj",
        )?;
        Ok(adapted)
    }
}

/// Names mirror `fsq_layer.{in_proj,out_proj}.*` — the checkpoint prefix
/// (`fsq_layer`) is added by the top-level [`crate::model::audio::voxcpm::model::VoxCpm2Model`]
/// composition, not here. Both projections may enumerate empty (a
/// block-quantized `in_proj`/`out_proj` has no `Var<R>` weight) — see
/// [`MaybeLoraLinear::parameters`](crate::nn::maybe_lora::MaybeLoraLinear).
impl<R: Runtime<DType = DType>> Module<R> for ScalarQuantization<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.in_proj);
        params.extend(child_params(&self.out_proj));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "in_proj", self.in_proj.named_parameters());
        extend_named(&mut params, "out_proj", self.out_proj.named_parameters());
        params
    }
}

/// The six auxiliary projections around `fsq_layer` that a future
/// `VoxCpm2Model` orchestrator will own: encoder/DiT bridges and the stop
/// classifier. See [`crate::model::audio::voxcpm::fsq::loader`] for the
/// checkpoint key layout each field is loaded from.
///
/// All six are [`MaybeLoraLinear`] for the same reason
/// [`ScalarQuantization`]'s pair is: a GGUF stores them block-quantized and
/// they multiply PACKED, while a safetensors checkpoint yields the
/// `Standard` variant and the dense path is unchanged. `MaybeLoraLinear`
/// additionally lets any of the six carry a LoRA adapter.
pub struct AuxProjections<R: Runtime> {
    pub enc_to_lm_proj: MaybeLoraLinear<R>,
    pub lm_to_dit_proj: MaybeLoraLinear<R>,
    pub res_to_dit_proj: MaybeLoraLinear<R>,
    pub fusion_concat_proj: MaybeLoraLinear<R>,
    pub stop_proj: MaybeLoraLinear<R>,
    /// Bias-free: the checkpoint carries no `stop_head.bias` tensor. See
    /// [`crate::model::audio::voxcpm::fsq::loader`] for how this is loaded.
    pub stop_head: MaybeLoraLinear<R>,
}

impl<R: Runtime<DType = DType>> AuxProjections<R> {
    /// `stop_head(silu(stop_proj(hidden)))`: the fixed composition the
    /// reference always runs together to produce stop-token logits.
    pub fn stop<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        // The extra three bounds over a dense `Linear::forward` — see
        // [`ScalarQuantization::forward`].
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R>,
    {
        let projected = self.stop_proj.forward(client, hidden)?;
        let activated = var_silu(&projected, client).map_err(Error::Numr)?;
        self.stop_head.forward(client, &activated)
    }

    /// Wrap any of the six projections that `targets` names with a fresh
    /// LoRA adapter, returning how many were adapted. `prefix` is passed
    /// straight through with NO segment appended — these six live at the
    /// checkpoint ROOT with no shared prefix (see the struct doc and
    /// [`Module::named_parameters`] above), so the owning
    /// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)
    /// calls this the same way it calls `named_parameters` on `aux`: with
    /// whatever prefix IT was itself given, unchanged. A leaf step: no
    /// zero-match check here — see
    /// [`crate::model::audio::voxcpm::minicpm4::MiniCpm4Attention::apply_lora`]'s
    /// doc comment for why.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = adapt_if_targeted(
            &mut self.enc_to_lm_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "enc_to_lm_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.lm_to_dit_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "lm_to_dit_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.res_to_dit_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "res_to_dit_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.fusion_concat_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "fusion_concat_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.stop_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "stop_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.stop_head,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "stop_head",
        )?;
        Ok(adapted)
    }
}

/// Names ARE the checkpoint root-level keys verbatim (`enc_to_lm_proj`,
/// `lm_to_dit_proj`, `res_to_dit_proj`, `fusion_concat_proj`, `stop_proj`,
/// `stop_head`) — these six live at the checkpoint root with no shared
/// prefix (see [`crate::model::audio::voxcpm::fsq::loader`]), so the
/// top-level [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)
/// composition adds NO prefix here, unlike every other sub-model.
impl<R: Runtime<DType = DType>> Module<R> for AuxProjections<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.enc_to_lm_proj);
        params.extend(child_params(&self.lm_to_dit_proj));
        params.extend(child_params(&self.res_to_dit_proj));
        params.extend(child_params(&self.fusion_concat_proj));
        params.extend(child_params(&self.stop_proj));
        params.extend(child_params(&self.stop_head));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "enc_to_lm_proj",
            self.enc_to_lm_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "lm_to_dit_proj",
            self.lm_to_dit_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "res_to_dit_proj",
            self.res_to_dit_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "fusion_concat_proj",
            self.fusion_concat_proj.named_parameters(),
        );
        extend_named(&mut params, "stop_proj", self.stop_proj.named_parameters());
        extend_named(&mut params, "stop_head", self.stop_head.named_parameters());
        params
    }
}

#[cfg(test)]
mod tests;
