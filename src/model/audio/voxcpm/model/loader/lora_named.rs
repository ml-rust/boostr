//! [`VoxCpm2Model::load_lora_named`] — split out of `loader.rs` to keep it
//! under the crate's 500-line hard limit for model-architecture files.

use super::VoxCpm2Model;
use crate::error::Result;
use crate::nn::{Module, named_tensors_to_id_map};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Load a LoRA adapter saved to safetensors — keyed by NAME, not by
    /// [`TensorId`](numr::tensor::TensorId) — and write it into this
    /// model's adapters.
    ///
    /// [`Self::load_lora_parameters`] exists for the in-PROCESS path:
    /// `SimpleTrainer::step`'s `TensorId`-keyed map, built from the SAME
    /// `Var`s this model still holds. A saved adapter file carries no such
    /// map — `TensorId`s are minted fresh by
    /// [`LoraLinear::new`](crate::nn::LoraLinear::new) every time
    /// [`Self::apply_lora`] runs, so they mean nothing across a save/load
    /// boundary or a second process. This function is the bridge: it reads
    /// each name off `self.named_parameters()`, resolves it to the `Var`'s
    /// current `TensorId`, and delegates to `load_lora_parameters`.
    ///
    /// # `apply_lora` MUST run first
    ///
    /// `named_parameters()` only lists `lora_a`/`lora_b` entries for
    /// projections that already carry an adapter — before `apply_lora`,
    /// every LoRA-suffixed name this function looks for is simply absent.
    /// Calling this on an unadapted model would then be a silent no-op:
    /// zero adapter names to resolve, so nothing to write, and `Ok(0)`
    /// looks identical to "the file matched everything". `tensors`
    /// non-empty against zero resolvable names is exactly the "stale/extra
    /// key" case below, so it errors instead — that is the failure this
    /// function exists to make impossible, not a case it degrades into.
    ///
    /// # Errors
    ///
    /// Hard-errors, never silently skips, on:
    /// - a `tensors` key matching no `lora_a`/`lora_b` `Var` in the model
    ///   (stale/extra key — the wrong `--targets` case)
    /// - an adapter `Var` with no matching key in `tensors` (missing key —
    ///   a partial adapter file)
    /// - a shape mismatch between a `Var` and its incoming tensor (the
    ///   wrong `--rank` case)
    ///
    /// Returns the number of adapter tensors written (2 per adapted
    /// projection present in `tensors`).
    pub fn load_lora_named(&mut self, tensors: &HashMap<String, Tensor<R>>) -> Result<usize> {
        let adapter_vars: Vec<(String, &Var<R>)> = self
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        let by_id = named_tensors_to_id_map(&adapter_vars, tensors)?;
        self.load_lora_parameters(&by_id)
    }
}

#[cfg(test)]
mod tests;
