//! [`VoxCpm2Model::load_lora_adapter`] — apply a saved LoRA adapter file
//! using ITS OWN rank/alpha/targets, read from the file's `__metadata__`,
//! rather than a caller-supplied `--rank`/`--alpha`/`--targets` that can
//! drift out of sync with what was actually trained. Split out of
//! `loader.rs` to keep it under the crate's 500-line hard limit for
//! model-architecture files, same as `lora_named.rs`.

use super::VoxCpm2Model;
use crate::error::{Error, Result};
use crate::format::SafeTensors;
use crate::nn::{LoraTargets, parse_lora_metadata};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::path::Path;

/// What [`VoxCpm2Model::load_lora_adapter`] read from and did with one
/// adapter file — for a caller (e.g. blazr) to log without reaching into
/// the model.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterReport {
    /// LoRA rank, read from the file's `lora_rank` metadata.
    pub rank: usize,
    /// LoRA alpha, read from the file's `lora_alpha` metadata.
    pub alpha: f32,
    /// Target projection names, read from the file's `lora_targets`
    /// metadata.
    pub targets: Vec<String>,
    /// Projections wrapped with a LoRA adapter (`apply_lora`'s count).
    pub projections_adapted: usize,
    /// Adapter tensors written into the model (`load_lora_named`'s count).
    pub tensors_loaded: usize,
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Apply a LoRA adapter saved to safetensors at `path`, using the
    /// rank/alpha/targets carried in ITS OWN `__metadata__` — no
    /// caller-supplied config to mismatch against what was trained.
    ///
    /// Runs the same sequence `examples/voxcpm/lora_load.rs`'s
    /// `load_lora_adapter` hand-wires for the CLI tools: open the file,
    /// parse its metadata, [`Self::apply_lora`] to allocate the adapters,
    /// [`SafeTensors::load_all`] to read the tensors, then
    /// [`Self::load_lora_named`] to write them in by name.
    ///
    /// # Errors
    ///
    /// Errors naming `path` when: the file cannot be opened; it carries no
    /// `__metadata__`, or is missing/has a malformed `lora_rank`,
    /// `lora_alpha`, or `lora_targets` key — meaning it was not written by
    /// `voxcpm_finetune`; `apply_lora` rejects a target (no such
    /// projection); or `load_lora_named` finds a stale/missing tensor key
    /// or a shape mismatch.
    pub fn load_lora_adapter(
        &mut self,
        path: &Path,
        device: &R::Device,
    ) -> Result<LoraAdapterReport> {
        let mut adapter = SafeTensors::open(path).map_err(|e| Error::ModelError {
            reason: format!(
                "--lora {}: failed to open adapter file: {e}",
                path.display()
            ),
        })?;
        let meta = parse_lora_metadata(adapter.metadata()).map_err(|e| Error::ModelError {
            reason: format!(
                "--lora {}: {e}; this file was not written by voxcpm_finetune, or lacks \
                 lora_rank/lora_alpha/lora_targets",
                path.display()
            ),
        })?;

        let targets = LoraTargets::new(meta.targets.clone());
        // `apply_lora` must run before `load_lora_named`: it allocates the
        // `lora_a`/`lora_b` Vars that name lookup then resolves against —
        // same order `examples/voxcpm/lora_load.rs` uses.
        let projections_adapted = self.apply_lora(&targets, meta.rank, meta.alpha, device)?;
        let tensors = adapter.load_all::<R>(device)?;
        let tensors_loaded = self.load_lora_named(&tensors)?;

        // A file-loaded adapter is a FROZEN artifact for inference, never a
        // fresh warm-start: `apply_lora` mints `lora_a`/`lora_b` with
        // `requires_grad = true`, so every forward would keep building an
        // autograd graph nobody ever calls `backward` on. Dropping that
        // graph after each render recurses over every recorded node, deep
        // enough to overflow a bounded worker stack (observed in blazr: a
        // 2 MB tokio worker overflowed on drop). Freeze here; a caller that
        // resumes fine-tuning this adapter calls `set_lora_trainable(true)`
        // itself right after (see `examples/voxcpm/finetune.rs`'s `--lora`
        // warm-start arm).
        self.set_lora_trainable(false);

        Ok(LoraAdapterReport {
            rank: meta.rank,
            alpha: meta.alpha,
            targets: meta.targets,
            projections_adapted,
            tensors_loaded,
        })
    }

    /// Set every `lora_a`/`lora_b` this model currently holds — across
    /// `feat_encoder`, `base_lm`, `residual_lm`, `feat_decoder`, `fsq`, and
    /// `aux` — to `trainable`, returning the total number of adapted
    /// projections touched. Mirrors [`Self::apply_lora`]'s walk exactly
    /// (same six sub-models, same order) but needs no `targets`/`prefix`/
    /// validation: this is a blanket toggle over whatever `apply_lora`
    /// already adapted, not a fresh name match.
    pub fn set_lora_trainable(&mut self, trainable: bool) -> usize {
        let mut touched = self.feat_encoder.set_lora_trainable(trainable);
        touched += self.base_lm.set_lora_trainable(trainable);
        touched += self.residual_lm.set_lora_trainable(trainable);
        touched += self.feat_decoder.set_lora_trainable(trainable);
        touched += self.fsq.set_lora_trainable(trainable);
        touched += self.aux.set_lora_trainable(trainable);
        touched
    }
}

#[cfg(test)]
mod tests {
    //! [`super::VoxCpm2Model::load_lora_adapter`] must reproduce, on a FRESH
    //! model, exactly what hand-wiring `apply_lora` + `load_lora_named` against
    //! a saved adapter file produces — same projections adapted, same tensors
    //! loaded, same forward output — while reading rank/alpha/targets from the
    //! file's own `__metadata__` instead of a caller-supplied config.
    //!
    //! Uses the tiny full [`VoxCpm2Model`] fixture from `generate::tests`
    //! (`fixture`/`model`), the same one `prefill.rs`'s inline tests use for
    //! the no-reference prefill path — real enough to run `prefill`, cheap
    //! enough to build twice per test.

    use super::*;
    use crate::model::audio::voxcpm::model::config::AUDIO_START_ID;
    use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, model};
    use crate::nn::{LoraTargets, Module, build_lora_metadata};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    const RANK: usize = 2;
    const ALPHA: f32 = 4.0;

    fn target_names() -> Vec<String> {
        vec!["q_proj".to_string(), "v_proj".to_string()]
    }

    /// Every `lora_a`/`lora_b` currently on `model`, as CPU tensors keyed by
    /// their full checkpoint-style path — the same shape
    /// `examples/voxcpm/finetune.rs::collect_adapter_tensors` saves.
    fn collect_adapter_tensors(
        model: &VoxCpm2Model<CpuRuntime>,
    ) -> HashMap<String, numr::tensor::Tensor<CpuRuntime>> {
        let mut out = HashMap::new();
        for (name, var) in Module::named_parameters(model) {
            if name.ends_with("lora_a") || name.ends_with("lora_b") {
                out.insert(name, var.tensor().contiguous().expect("contiguous"));
            }
        }
        out
    }

    #[test]
    fn load_lora_adapter_matches_hand_applied_model_and_reports_metadata() {
        let (client, device) = cpu_setup();

        // Hand-applied reference: apply_lora directly, no file round-trip.
        let mut hand_applied = model(fixture(false, &device), &device);
        let targets = LoraTargets::new(target_names());
        let hand_adapted = hand_applied
            .apply_lora(&targets, RANK, ALPHA, &device)
            .expect("apply_lora");
        assert!(
            hand_adapted > 0,
            "q_proj/v_proj must match at least one projection"
        );

        let adapter_tensors = collect_adapter_tensors(&hand_applied);
        let metadata = build_lora_metadata(RANK, ALPHA, &target_names());

        let file = NamedTempFile::new().expect("tempfile");
        crate::format::safetensors::save_safetensors(
            file.path(),
            &adapter_tensors,
            Some(&metadata),
        )
        .expect("save adapter");

        // Fresh, unadapted model: load_lora_adapter must derive rank/alpha/
        // targets from the file alone.
        let mut loaded = model(fixture(false, &device), &device);
        let report = loaded
            .load_lora_adapter(file.path(), &device)
            .expect("load_lora_adapter");

        assert_eq!(report.rank, RANK);
        assert_eq!(report.alpha, ALPHA);
        let mut found_targets = report.targets.clone();
        found_targets.sort();
        assert_eq!(found_targets, target_names());
        assert_eq!(report.projections_adapted, hand_adapted);
        assert_eq!(report.tensors_loaded, adapter_tensors.len());

        let text_token_ids = [11u32, 22, AUDIO_START_ID];
        let hand_prefill = hand_applied
            .prefill(&client, None, &text_token_ids, text_token_ids.len())
            .expect("hand-applied prefill");
        let loaded_prefill = loaded
            .prefill(&client, None, &text_token_ids, text_token_ids.len())
            .expect("loaded prefill");

        assert_eq!(
            hand_prefill.lm_hidden.tensor().to_vec::<f32>(),
            loaded_prefill.lm_hidden.tensor().to_vec::<f32>(),
            "load_lora_adapter must reproduce the hand-applied model's lm_hidden"
        );
        assert_eq!(
            hand_prefill.residual_hidden.tensor().to_vec::<f32>(),
            loaded_prefill.residual_hidden.tensor().to_vec::<f32>(),
            "load_lora_adapter must reproduce the hand-applied model's residual_hidden"
        );

        // The actual fix: `load_lora_adapter` must freeze the adapter it
        // just loaded, so an inference forward records no autograd graph.
        // `hand_applied` never freezes (apply_lora alone leaves
        // `requires_grad = true`), so contrasting the two proves this is
        // `load_lora_adapter`'s own behavior, not a property of `prefill`.
        assert!(
            hand_prefill.lm_hidden.requires_grad(),
            "sanity: the hand-applied (un-frozen) model's prefill must still track grad"
        );
        assert!(
            !loaded_prefill.lm_hidden.requires_grad(),
            "load_lora_adapter must freeze the loaded adapter — prefill must not require grad"
        );
    }

    #[test]
    fn refuses_a_safetensors_file_with_no_metadata() {
        let (_client, device) = cpu_setup();
        let mut loaded = model(fixture(false, &device), &device);

        let tensors: HashMap<String, numr::tensor::Tensor<CpuRuntime>> = HashMap::new();
        let file = NamedTempFile::new().expect("tempfile");
        crate::format::safetensors::save_safetensors(file.path(), &tensors, None)
            .expect("save empty adapter");

        let err = loaded
            .load_lora_adapter(file.path(), &device)
            .expect_err("no metadata must be refused");
        let message = err.to_string();
        assert!(
            message.contains(&file.path().display().to_string()),
            "error must name the path: {message}"
        );
        assert!(
            message.contains("voxcpm_finetune"),
            "error must say the file was not written by voxcpm_finetune: {message}"
        );
    }
}
