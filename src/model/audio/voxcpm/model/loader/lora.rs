//! Whole-model LoRA: adapter attachment across every sub-model, the
//! structural projection path list it validates against, and optimizer
//! write-back.

use super::model::VoxCpm2Model;
use crate::error::Result;
use crate::model::audio::voxcpm::fsq::loader::FSQ_LAYER_PREFIX;
use crate::model::audio::voxcpm::local_dit::DEFAULT_LOCAL_DIT_PREFIX;
use crate::model::audio::voxcpm::local_encoder::DEFAULT_LOCAL_ENCODER_PREFIX;
use crate::model::audio::voxcpm::minicpm4::{DEFAULT_MINICPM4_PREFIX, DEFAULT_RESIDUAL_LM_PREFIX};
use crate::nn::LoraTargets;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// THE entry point: wrap every `targets`-named projection across the
    /// whole model — `feat_encoder`, `base_lm`, `residual_lm`,
    /// `feat_decoder`, `fsq_layer`, and `aux`'s six root-level projections —
    /// with a fresh LoRA adapter, so a fine-tune can train adapters over the
    /// frozen base (every VoxCPM2 weight loads `requires_grad = false`; see
    /// `local_encoder/encoder.rs:19`, `minicpm4/model.rs:30`). Returns the
    /// total number of projections adapted.
    ///
    /// `vae_encoder`/`vae_decoder` are never touched: they are a separately
    /// checkpointed, frozen audio codec (see the module docs) and neither
    /// implements `Module<R>` — same exclusion [`crate::nn::Module::named_parameters`]
    /// documents.
    ///
    /// Delegates to each sub-model's own `apply_lora`, joining ITS prefix
    /// with the same constant [`crate::nn::Module::named_parameters`] uses below
    /// (`DEFAULT_LOCAL_ENCODER_PREFIX`, `DEFAULT_MINICPM4_PREFIX`, ...) so a
    /// target's full path here always matches the same-named path
    /// `named_parameters()` would produce — the two are never built by
    /// separately hand-written logic.
    ///
    /// As the actual top of the call graph, this is the ONE place that
    /// validates every target up front with [`LoraTargets::ensure_all_match`]
    /// against the WHOLE model's candidate set — [`Self::lora_projection_names`],
    /// NOT `self.named_parameters()` — before adapting anything. Each
    /// sub-model child is then walked via its `apply_lora_unchecked`, not
    /// its validating `apply_lora`: re-validating a cross-subtree target
    /// list against only one child's own candidate set would reject a
    /// target that lives in a sibling (e.g. `stop_proj`, which lives under
    /// `aux`, is not a candidate inside `feat_encoder`'s own subtree).
    ///
    /// # Why the candidate set must be STRUCTURAL, not parameter-derived
    ///
    /// `named_parameters()` answers "which projections currently carry a
    /// dense `Var<R>`", which on a QUANTIZED (GGUF) checkpoint is nearly
    /// EMPTY: `MaybeQuantLinear::named_parameters()` returns nothing for a
    /// block-quantized projection (the weight has no `Var<R>`, only packed
    /// bytes `quant_matmul` reads directly). Measured on a real VoxCPM2
    /// GGUF, that shrank the candidate set from 577 (dense) to 131,
    /// rejecting a perfectly valid `["q_proj", "v_proj"]` target with
    /// "matched no projection by dot-segment name" — QLoRA's entire
    /// reason to exist is adapting a quantized base, so that checkpoint is
    /// exactly the one this validation must not reject.
    /// [`Self::lora_projection_names`] instead answers "which projections
    /// this tree's `MaybeLoraLinear` FIELDS structurally are", which is the
    /// same 577-projection set on every checkpoint dtype: dense,
    /// block-quantized, or decomposed-quantized. The wrapping itself was
    /// never the bug — [`LoraLinear::new`](crate::nn::LoraLinear::new) sizes
    /// its adapter from [`MaybeQuantLinear::shape`](crate::nn::MaybeQuantLinear::shape),
    /// which works on every variant — only this validation's candidate
    /// source was.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
    ) -> Result<usize> {
        let candidates = self.lora_projection_names();
        targets.ensure_all_match(&candidates)?;

        let mut adapted = self.feat_encoder.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_LOCAL_ENCODER_PREFIX,
        )?;
        adapted += self.base_lm.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_MINICPM4_PREFIX,
        )?;
        adapted += self.residual_lm.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_RESIDUAL_LM_PREFIX,
        )?;
        adapted += self.feat_decoder.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_LOCAL_DIT_PREFIX,
        )?;
        adapted += self
            .fsq
            .apply_lora(targets, rank, alpha, device, FSQ_LAYER_PREFIX)?;
        // Root-level, no prefix — see `Module::named_parameters` above.
        adapted += self.aux.apply_lora(targets, rank, alpha, device, "")?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt across
    /// the WHOLE model — `feat_encoder`, `base_lm`, `residual_lm`,
    /// `feat_decoder`, `fsq_layer`, and `aux`'s six root-level projections —
    /// INDEPENDENT of whether any of them is dense, block-quantized, or
    /// decomposed-quantized. See [`Self::apply_lora`]'s doc comment for WHY
    /// this must be structural rather than parameter-derived (the GGUF
    /// case).
    ///
    /// Delegates to each sub-model's own `lora_projection_names`, joined at
    /// the SAME prefix constants [`Self::apply_lora`] passes to that same
    /// sub-model's `apply_lora_unchecked` — `DEFAULT_LOCAL_ENCODER_PREFIX`,
    /// `DEFAULT_MINICPM4_PREFIX`, `DEFAULT_RESIDUAL_LM_PREFIX`,
    /// `DEFAULT_LOCAL_DIT_PREFIX`, `FSQ_LAYER_PREFIX`, and `aux`'s bare `""`
    /// — so a path here is never built by separately hand-written logic:
    /// [`Self::apply_lora`] and this walk read the SAME constants in the
    /// SAME order, the only difference being `_unchecked`'s mutable adapt
    /// vs. this method's read-only name collection.
    pub fn lora_projection_names(&self) -> Vec<String> {
        let mut names = self
            .feat_encoder
            .lora_projection_names(DEFAULT_LOCAL_ENCODER_PREFIX);
        names.extend(self.base_lm.lora_projection_names(DEFAULT_MINICPM4_PREFIX));
        names.extend(
            self.residual_lm
                .lora_projection_names(DEFAULT_RESIDUAL_LM_PREFIX),
        );
        names.extend(
            self.feat_decoder
                .lora_projection_names(DEFAULT_LOCAL_DIT_PREFIX),
        );
        names.extend(self.fsq.lora_projection_names(FSQ_LAYER_PREFIX));
        // Root-level, no prefix — see `Module::named_parameters` above.
        names.extend(self.aux.lora_projection_names(""));
        names
    }

    /// THE write-back entry point: apply an optimizer's updated adapter
    /// tensors — keyed by [`TensorId`], e.g.
    /// [`SimpleTrainer::step`](crate::trainer::simple::SimpleTrainer::step)'s
    /// output — back onto every `MaybeLoraLinear` adapter across the whole
    /// model, in place, preserving each adapter's `TensorId`.
    ///
    /// Without this, a training loop that calls `backward` then
    /// `SimpleTrainer::step` never actually updates the model: `step` writes
    /// into the `HashMap<TensorId, Tensor<R>>` it returns, but the `Var`s
    /// this model's `MaybeLoraLinear`s hold are untouched by that write, so
    /// every subsequent forward pass would keep recomputing from the SAME
    /// pre-update weights.
    ///
    /// Unlike [`Self::apply_lora`], this needs no `targets`/`prefix`
    /// threading and no [`LoraTargets::ensure_all_match`] validation:
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] looks each
    /// adapter up by its own stable `TensorId`, which is already unique —
    /// there is no dotted path to match against and no zero-match trap to
    /// guard. Returns the total number of adapter TENSORS written (2 per
    /// adapted projection whose ids are both present in `params`).
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = self.feat_encoder.load_lora_parameters(params)?;
        written += self.base_lm.load_lora_parameters(params)?;
        written += self.residual_lm.load_lora_parameters(params)?;
        written += self.feat_decoder.load_lora_parameters(params)?;
        written += self.fsq.load_lora_parameters(params)?;
        written += self.aux.load_lora_parameters(params)?;
        Ok(written)
    }
}
