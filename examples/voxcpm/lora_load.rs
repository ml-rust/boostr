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
use boostr::nn::check_lora_metadata;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Check `lora_path`'s metadata against the flags given, then apply it via
/// [`VoxCpm2Model::load_lora_adapter`] (the src-level entry point, which
/// applies the adapter using ITS OWN metadata) — the exact sequence
/// `voxcpm_clone`'s `--lora` arm uses. Shared by `--eval-only` (score base +
/// adapter) and training (warm-start the adapters from the file) — see
/// `voxcpm_finetune`'s module docs' "`--lora`" section.
///
/// `check_lora_metadata` runs BEFORE `load_lora_adapter`: a
/// rank/alpha/targets mismatch against the flags given here must abort
/// before any `Var` is allocated, not after. Its error already names the
/// disagreeing field and both values; this wraps it with the file path, the
/// flag values given, and the fix, since the bare mismatch names neither.
///
/// Returns `(projections_adapted, tensors_loaded)`, both from
/// [`boostr::model::audio::voxcpm::model::LoraAdapterReport`], the same
/// counts `voxcpm_clone` logs.
pub fn load_lora_adapter<R: Runtime<DType = DType>>(
    model: &mut VoxCpm2Model<R>,
    lora_path: &Path,
    rank: usize,
    alpha: f32,
    target_names: &[String],
    device: &R::Device,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    eprintln!("loading LoRA adapter {} ...", lora_path.display());
    let adapter = SafeTensors::open(lora_path)?;
    check_lora_metadata(adapter.metadata(), rank, alpha, target_names).map_err(|e| {
        format!(
            "--lora {}: adapter does not match --rank {rank} --alpha {alpha} --targets \
             {target_names:?}: {e}; fix: pass the --rank/--alpha/--targets this adapter was \
             saved with, or retrain to match the ones given here",
            lora_path.display()
        )
    })?;
    let report = model.load_lora_adapter(lora_path, device)?;
    Ok((report.projections_adapted, report.tensors_loaded))
}
