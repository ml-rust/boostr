//! LoRA adapter loading, shared by `voxcpm_finetune` and `voxcpm_clone`.
//!
//! Both binaries must load a saved adapter file into a model identically:
//! `voxcpm_clone --lora` reconstructs the exact eval `voxcpm_finetune
//! --eval-only --lora` scored, and a divergence in load order or in which
//! metadata check runs first would make a clone's output not reproduce a
//! finetune run's own eval.

use std::path::Path;

use boostr::format::SafeTensors;
use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::nn::{LoraTargets, check_lora_metadata};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Apply LoRA to `model` and load a saved adapter's tensors into it — the
/// exact sequence and functions `voxcpm_clone`'s `--lora` arm uses
/// (`check_lora_metadata`, `SafeTensors::open`/`load_all`,
/// `load_lora_named`), reused here so both binaries load one file
/// identically. Shared by `--eval-only` (score base + adapter) and training
/// (warm-start the adapters from the file) — see `voxcpm_finetune`'s module
/// docs' "`--lora`" section.
///
/// `check_lora_metadata` runs BEFORE `apply_lora`: a rank/alpha/targets
/// mismatch must abort before any `Var` is allocated, not after. Its error
/// already names the disagreeing field and both values; this wraps it with
/// the file path, the flag values given, and the fix, since the bare
/// mismatch names neither.
///
/// Returns `(projections_adapted, tensors_loaded)`, both from the same calls
/// `voxcpm_clone` logs.
pub fn load_lora_adapter<R: Runtime<DType = DType>>(
    model: &mut VoxCpm2Model<R>,
    lora_path: &Path,
    rank: usize,
    alpha: f32,
    target_names: &[String],
    device: &R::Device,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    eprintln!("loading LoRA adapter {} ...", lora_path.display());
    let mut adapter = SafeTensors::open(lora_path)?;
    check_lora_metadata(adapter.metadata(), rank, alpha, target_names).map_err(|e| {
        format!(
            "--lora {}: adapter does not match --rank {rank} --alpha {alpha} --targets \
             {target_names:?}: {e}; fix: pass the --rank/--alpha/--targets this adapter was \
             saved with, or retrain to match the ones given here",
            lora_path.display()
        )
    })?;
    let lora_targets = LoraTargets::new(target_names.to_vec());
    // `apply_lora` must run before `load_lora_named`: it allocates the
    // `lora_a`/`lora_b` Vars that name lookup then resolves against — same
    // order `voxcpm_clone`'s --lora arm uses.
    let adapted = model.apply_lora(&lora_targets, rank, alpha, device)?;
    let lora_tensors = adapter.load_all::<R>(device)?;
    let loaded = model.load_lora_named(&lora_tensors)?;
    Ok((adapted, loaded))
}
