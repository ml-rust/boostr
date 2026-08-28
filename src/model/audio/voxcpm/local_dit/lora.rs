//! LoRA adaptation for [`LocalDit`] — split out of `loader.rs` to stay under
//! the crate's 500-line model-architecture file limit.

use crate::error::Result;
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::nn::{LoraTargets, adapt_if_targeted, load_lora_child, push_projection_name};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// Wrap `in_proj`/`cond_proj`/`out_proj`, `time_mlp`/`delta_time_mlp`,
    /// and every layer's targeted projections with a fresh LoRA adapter,
    /// returning the total adapted. `prefix` mirrors
    /// `Module::named_parameters` (see `loader.rs`) exactly: `"estimator.in_proj"`
    /// etc. are joined straight onto `prefix`, and each layer is joined at
    /// `"estimator.decoder.layers.{i}"`.
    ///
    /// This is the entry point for adapting this sub-model DIRECTLY (a
    /// caller may adapt just `feat_decoder` on its own), so it validates
    /// every target up front with [`LoraTargets::ensure_all_match`] against
    /// this tree's OWN full candidate set — [`Self::lora_projection_names`],
    /// NOT `self.named_parameters()` — before delegating to
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
    /// `prefix` — `estimator.in_proj`/`estimator.cond_proj`/
    /// `estimator.out_proj`, `estimator.time_mlp`/`estimator.delta_time_mlp`,
    /// plus every layer's projections — INDEPENDENT of whether any of them
    /// is dense, block-quantized, or decomposed-quantized. This is what
    /// fixes the QLoRA validation bug: on a GGUF checkpoint every
    /// projection here is block-quantized, so `named_parameters()` returns
    /// EMPTY for all of them and a valid target would be rejected as
    /// matching nothing. `norm`/`rope`/`time_embeddings` carry no
    /// [`crate::nn::MaybeLoraLinear`] projections, so none contributes a
    /// name. Matches [`Self::apply_lora_unchecked`]'s walk exactly: each
    /// name is joined via the SAME [`push_projection_name`] helper or the
    /// SAME [`LoraTargets::join`]-built prefix `apply_lora_unchecked` passes
    /// to each child, so a path here is never built by separately
    /// hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        push_projection_name(&mut names, prefix, "estimator.in_proj");
        push_projection_name(&mut names, prefix, "estimator.cond_proj");
        push_projection_name(&mut names, prefix, "estimator.out_proj");
        names.extend(
            self.time_mlp
                .lora_projection_names(&LoraTargets::join(prefix, "estimator.time_mlp")),
        );
        names.extend(
            self.delta_time_mlp
                .lora_projection_names(&LoraTargets::join(prefix, "estimator.delta_time_mlp")),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            names.extend(layer.lora_projection_names(&LoraTargets::join(
                prefix,
                &format!("estimator.decoder.layers.{i}"),
            )));
        }
        names
    }

    /// Same walk as [`Self::apply_lora`] but skips
    /// [`LoraTargets::ensure_all_match`]. Exists for a parent
    /// (`VoxCpm2Model`) that has already validated `targets` against the
    /// WHOLE model: re-validating here against only this subtree would
    /// reject a target that lives in a sibling (`feat_encoder`, `base_lm`,
    /// `residual_lm`, `aux`), even though it is perfectly valid at root.
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
            "estimator.in_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.cond_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "estimator.cond_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.out_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "estimator.out_proj",
        )?;
        adapted += self.time_mlp.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "estimator.time_mlp"),
        )?;
        adapted += self.delta_time_mlp.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "estimator.delta_time_mlp"),
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            adapted += layer.apply_lora(
                targets,
                rank,
                alpha,
                device,
                &LoraTargets::join(prefix, &format!("estimator.decoder.layers.{i}")),
            )?;
        }
        Ok(adapted)
    }

    /// Write back updated adapter values for `in_proj`/`cond_proj`/`out_proj`,
    /// `time_mlp`/`delta_time_mlp`, and every layer from an optimizer's
    /// `params` map, keeping every adapter's [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix or target validation needed here
    /// — unlike [`Self::apply_lora`], lookup is by ID, not by dotted path.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.in_proj, params, "estimator.in_proj")?;
        written += load_lora_child(&mut self.cond_proj, params, "estimator.cond_proj")?;
        written += load_lora_child(&mut self.out_proj, params, "estimator.out_proj")?;
        written += self.time_mlp.load_lora_parameters(params)?;
        written += self.delta_time_mlp.load_lora_parameters(params)?;
        for layer in self.layers.iter_mut() {
            written += layer.load_lora_parameters(params)?;
        }
        Ok(written)
    }
}
