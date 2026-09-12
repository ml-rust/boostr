//! LoRA adaptation for [`MiniCpm4Attention`]: adapter attachment, projection
//! path enumeration, optimizer write-back, and the trainable toggle.

use super::MiniCpm4Attention;
use crate::error::Result;
use crate::nn::{LoraTargets, adapt_if_targeted, load_lora_child, push_projection_name};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> MiniCpm4Attention<R> {
    /// Wrap `q_proj`/`k_proj`/`v_proj`/`o_proj` that `targets` names with a
    /// fresh LoRA adapter each, returning how many were adapted.
    ///
    /// `prefix` is the dotted path the OWNING [`MiniCpm4Layer`](crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer)
    /// would pass to [`crate::nn::extend_named`] for this block — `"self_attn"` under
    /// `MiniCpm4Layer::named_parameters`'s own convention — so each
    /// projection's full path here (via [`LoraTargets::join`]) matches
    /// `named_parameters()`'s path for that same projection exactly.
    ///
    /// This is a LEAF step in the bottom-up composition: it does NOT call
    /// [`LoraTargets::ensure_all_match`] itself, only the model-level entry
    /// points do (see their doc comments) — a target this block has no
    /// projection for (e.g. `gate_proj`) is not an error here, only if it
    /// matches nothing ANYWHERE in the tree the caller actually entered on.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = adapt_if_targeted(
            &mut self.q_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "q_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.k_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "k_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.v_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "v_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.o_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "o_proj",
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — `q_proj`, `k_proj`, `v_proj`, `o_proj` — INDEPENDENT of
    /// whether a projection is dense, block-quantized, or
    /// decomposed-quantized. This is what fixes the QLoRA validation bug: a
    /// GGUF-loaded `MiniCpm4Attention` has `named_parameters()` return
    /// EMPTY for every projection here (block-quantized storage has no
    /// `Var<R>`), so validating against `named_parameters()` rejects a
    /// perfectly valid `q_proj`/`v_proj` target on a quantized checkpoint.
    /// Which projections exist is a STRUCTURAL property of this type, not a
    /// function of whether its weights happen to be dense. Built with the
    /// same [`crate::nn::push_projection_name`] helper `apply_lora`'s
    /// [`adapt_if_targeted`] calls use, so a path here is never hand-written
    /// separately from the one `apply_lora` matches.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        push_projection_name(&mut names, prefix, "q_proj");
        push_projection_name(&mut names, prefix, "k_proj");
        push_projection_name(&mut names, prefix, "v_proj");
        push_projection_name(&mut names, prefix, "o_proj");
        names
    }

    /// Write back updated `q_proj`/`k_proj`/`v_proj`/`o_proj` adapter values
    /// from an optimizer's `params` map, keeping their [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix needed — unlike
    /// [`Self::apply_lora`], lookup is by ID.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.q_proj, params, "q_proj")?;
        written += load_lora_child(&mut self.k_proj, params, "k_proj")?;
        written += load_lora_child(&mut self.v_proj, params, "v_proj")?;
        written += load_lora_child(&mut self.o_proj, params, "o_proj")?;
        Ok(written)
    }

    /// Set every attached adapter's `lora_a`/`lora_b` to `trainable`,
    /// returning how many of the four projections carry an adapter. Unlike
    /// [`Self::apply_lora`], this touches only ALREADY-adapted projections —
    /// there is nothing to freeze/unfreeze on a `Plain` one — and needs no
    /// `targets`/`prefix`: it is a blanket toggle, not a name match.
    pub fn set_lora_trainable(&mut self, trainable: bool) -> usize {
        let mut touched = 0;
        for proj in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ] {
            if proj.is_adapted() {
                proj.set_trainable(trainable);
                touched += 1;
            }
        }
        touched
    }
}

#[cfg(test)]
mod tests {
    use super::super::block::tests::tiny_attention;
    use super::*;
    use crate::test_utils::cpu_setup;

    /// `apply_lora` on a leaf attention block: matched targets get wrapped,
    /// unmatched fields stay `Plain`, and the count reflects exactly the
    /// matched set.
    #[test]
    fn apply_lora_wraps_only_targeted_projections() {
        let (_client, device) = cpu_setup();
        let mut attn = tiny_attention(false, &device);
        let targets = LoraTargets::new(["q_proj", "v_proj"]);

        let adapted = attn
            .apply_lora(&targets, 2, 4.0, &device, "self_attn")
            .expect("apply_lora");
        assert_eq!(adapted, 2);
        assert!(attn.q_proj.is_adapted());
        assert!(attn.v_proj.is_adapted());
        assert!(!attn.k_proj.is_adapted());
        assert!(!attn.o_proj.is_adapted());
    }

    /// Adapting an already-adapted projection errors rather than silently
    /// discarding the existing adapter.
    #[test]
    fn apply_lora_rejects_double_adapt() {
        let (_client, device) = cpu_setup();
        let mut attn = tiny_attention(false, &device);
        let targets = LoraTargets::new(["q_proj"]);

        attn.apply_lora(&targets, 2, 4.0, &device, "self_attn")
            .expect("first apply_lora");
        let err = attn
            .apply_lora(&targets, 2, 4.0, &device, "self_attn")
            .unwrap_err();
        assert!(err.to_string().contains("already carries"), "got {err}");
    }
}
