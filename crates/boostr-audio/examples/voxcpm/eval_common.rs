//! Shared eval machinery for the VoxCPM2 examples: manifest reading, the
//! `--max-patches` filter, the per-row prefill/target build, and the fixed
//! teacher-forced CFM + stop loss the eval batch is scored with.
//!
//! `finetune.rs` owns the training loop and `sensitivity.rs` owns the
//! per-tensor perturbation sweep; both score the SAME loss over the SAME
//! rows with the SAME pinned `t`/noise, so the loss has exactly one
//! definition and it lives here. A second copy would let the two binaries
//! drift and would make their numbers incomparable, which is the whole
//! point of the measurement.
//!
//! Every determinism guarantee `finetune.rs`'s module docs state — the
//! `t`/noise draw off [`EVAL_NOISE_SEED`], the RNG-free row order and row
//! membership, `drop_cond = false`, the fixed summation order — is a
//! property of the code in THIS file, and holds identically for every
//! caller.

// Two example binaries compile this module separately, and each uses a
// subset of it. An item unused by one of them is not dead code.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use boostr::model::audio::voxcpm::model::config::{AUDIO_START_ID, VoxCpm2Config};
use boostr::model::audio::voxcpm::model::{PatchGenerator, VoxCpm2Model};
use boostr::model::audio::voxcpm::{PrefillState, VoxCpmClient};
use boostr::quant::traits::DequantOps;
use boostr_audio::voxcpm::{normalize_whitespace, tokenize};
use boostr_audio::{decode_audio, extension_hint, to_mono_at_rate};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::AnyTokenizer;

/// Rate every manifest wav is resampled to before the AudioVAE encoder.
/// Fixed by the encoder, not a choice — see `voxcpm_clone.rs`'s identical
/// constant.
pub const REF_RATE: u32 = 16_000;
/// `lambda_diff` in the reference VoxCPM `lambdas:` block. Fixed at the
/// reference implementation's own default — unlike `lambda_stop`, its FAQ names no failure mode
/// that calls for retuning it, so it is not exposed as a flag.
pub const LAMBDA_DIFF: f64 = 1.0;
/// Default `--max-patches`: retains 78% of targets on the measured real
/// corpus while staying under ~10 GB peak RSS — see `finetune.rs`'s module docs'
/// "Choosing `--max-patches`" table. Measured scaling is ~1.36 GB per second
/// of target audio (q6_k, CPU), so a 6.0 s cap keeps peak near `6.0 * 1.36
/// GB ≈ 8.2 GB` plus fixed model/runtime overhead — ~9.8 GB total. Converted
/// to patches at `patch_size = 4`
/// ([`VoxCpm2Config::default`](boostr::model::audio::voxcpm::model::config::VoxCpm2Config::default),
/// the checkpoint's usual value) and `HOP_LENGTH` (640): `6.0 s * 16_000
/// samples/s = 96_000 samples`; `ceil(96_000 / 640) = 150 frames`;
/// `ceil(150 / 4) = 38 patches`. A checkpoint with a different `patch_size`
/// shifts the seconds-per-patch ratio this default assumes, so pass
/// `--max-patches` explicitly for one.
pub const DEFAULT_MAX_PATCHES: usize = 38;
/// Default `--eval-rows`: how many of the kept manifest rows are held out
/// for the fixed eval batch — see `finetune.rs`'s module docs' "Eval batch" section.
pub const DEFAULT_EVAL_ROWS: usize = 4;
/// Seed for the eval batch's ONE draw of `t` and noise per row. Fixed here,
/// not derived from `--seed` or `step_counter`, so the eval metric is
/// comparable across runs that only differ in `--seed` — the whole point of
/// the eval batch is a number that moves with LEARNING, not with sampling.
pub const EVAL_NOISE_SEED: u64 = 0xE7A1_5EED;

/// One `(wav, text, ref_wav)` row resolved from the manifest.
pub struct ManifestRow {
    pub wav: PathBuf,
    pub text: String,
    /// The reference-conditioning clip, a DIFFERENT clip from the same
    /// speaker as `wav` — never `wav` itself. `None` when the manifest row
    /// left the (optional) `ref_wav` column empty or absent.
    pub ref_wav: Option<PathBuf>,
}

/// Resolve a manifest-relative wav path: absolute paths pass through,
/// everything else is joined onto `manifest_dir`.
pub fn resolve_wav_path(manifest_dir: &Path, field: &str) -> PathBuf {
    let path = PathBuf::from(field);
    if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    }
}

/// Parse a header-named TSV manifest: `wav` and `text` columns are required
/// and an optional `ref_wav` column, all located by NAME so extra columns
/// (speaker id, duration, ...) are ignored rather than rejected. A row with
/// no `ref_wav` value (empty cell, short row, or the column absent from the
/// header entirely) gets `ManifestRow::ref_wav == None`. `wav`/`ref_wav`
/// paths are resolved relative to the manifest's own directory when not
/// already absolute.
pub fn load_manifest(path: &Path) -> Result<Vec<ManifestRow>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: failed to read manifest: {e}", path.display()))?;
    let mut lines = contents.lines();

    let header = lines.next().ok_or_else(|| {
        format!(
            "{}: manifest is empty, expected a header row",
            path.display()
        )
    })?;
    let columns: Vec<&str> = header.split('\t').map(str::trim).collect();
    let wav_idx = columns.iter().position(|c| *c == "wav").ok_or_else(|| {
        format!(
            "{}: manifest header missing required column \"wav\" (found: {columns:?})",
            path.display()
        )
    })?;
    let text_idx = columns.iter().position(|c| *c == "text").ok_or_else(|| {
        format!(
            "{}: manifest header missing required column \"text\" (found: {columns:?})",
            path.display()
        )
    })?;
    // Optional: absent entirely means every row trains without a reference.
    let ref_wav_idx = columns.iter().position(|c| *c == "ref_wav");
    let needed = wav_idx.max(text_idx) + 1;

    let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut rows = Vec::new();
    for (offset, line) in lines.enumerate() {
        let line_no = offset + 2; // 1 for the header, 1 for 1-indexing
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < needed {
            return Err(format!(
                "{}:{line_no}: expected at least {needed} tab-separated column(s), got {}",
                path.display(),
                fields.len()
            )
            .into());
        }
        let wav_path = resolve_wav_path(manifest_dir, fields[wav_idx].trim());
        // A short row (ref_wav column present in the header but this line
        // has fewer fields) is the same as an empty cell: no reference.
        let ref_wav = ref_wav_idx
            .and_then(|idx| fields.get(idx))
            .map(|field| field.trim())
            .filter(|field| !field.is_empty())
            .map(|field| resolve_wav_path(manifest_dir, field));
        rows.push(ManifestRow {
            wav: wav_path,
            text: fields[text_idx].trim().to_string(),
            ref_wav,
        });
    }
    if rows.is_empty() {
        return Err(format!("{}: no data rows after the header", path.display()).into());
    }
    Ok(rows)
}

/// Patch count a clip of `samples` 16 kHz samples folds to, WITHOUT running
/// the AudioVAE encoder. `VoxCpm2Config::ref_pad_multiple` right-pads to a
/// multiple of `patch_size * HOP_LENGTH` before the real encode, so the true
/// frame count is always a multiple of `patch_size`; `ceil(ceil(samples /
/// HOP_LENGTH) / patch_size)` equals `ceil(samples / (patch_size *
/// HOP_LENGTH))`, exactly that padded-then-folded patch count — this is not
/// an approximation. Reuses [`VoxCpm2Config::ref_pad_multiple`] rather than
/// re-deriving `patch_size * HOP_LENGTH` here.
pub fn estimate_patches(samples: usize, cfg: &VoxCpm2Config) -> usize {
    samples.div_ceil(cfg.ref_pad_multiple())
}

/// Truncate a `ref_wav`'s 16 kHz samples to its leading
/// `max_patches * ref_pad_multiple()` samples — the same cap
/// [`estimate_patches`] checks against — so the AudioVAE encoder never sees
/// the excess. Safe ONLY for the reference clip, never the target `wav`:
/// see `finetune.rs`'s module docs' "Bounding training memory" section for why the two
/// are treated differently. A clip already at or under the cap is returned
/// unchanged.
pub fn truncate_reference(
    mut samples: Vec<f32>,
    cfg: &VoxCpm2Config,
    max_patches: usize,
) -> Vec<f32> {
    let cap_samples = max_patches * cfg.ref_pad_multiple();
    samples.truncate(cap_samples);
    samples
}

/// Filter `rows` to those whose target `wav` folds to at most `max_patches`
/// patches, without ever running the AudioVAE encoder — see the module
/// docs' "Bounding training memory" section. An over-cap `ref_wav` is NOT a
/// reason to drop the row — it is reported here as truncated (a separate
/// count from skipped rows) and the actual truncation happens where
/// `ref_wav` is loaded for training, via [`truncate_reference`]; this
/// function only measures and reports. Decodes each candidate clip once
/// here (cheap PCM decode, not the VAE) purely to read its sample count;
/// the training loop below decodes again per epoch, same as it always has.
/// Prints the kept rows' with-reference / no-reference split, the only
/// signal that a manifest lost its `ref_wav` column.
pub fn filter_rows_by_patch_cap<'a>(
    rows: &'a [ManifestRow],
    cfg: &VoxCpm2Config,
    max_patches: usize,
) -> Result<Vec<&'a ManifestRow>, Box<dyn std::error::Error>> {
    let mut kept = Vec::new();
    let mut skipped = 0usize;
    let mut truncated_refs = 0usize;
    let mut retained_seconds = 0.0f64;

    for row in rows {
        let wav = load_wav_16k(&row.wav).map_err(|e| format!("{}: {e}", row.wav.display()))?;
        let wav_patches = estimate_patches(wav.len(), cfg);
        if wav_patches > max_patches {
            eprintln!(
                "skip {}: {wav_patches} patches > --max-patches {max_patches}",
                row.wav.display()
            );
            skipped += 1;
            continue;
        }

        if let Some(ref_wav_path) = &row.ref_wav {
            let ref_wav = load_wav_16k(ref_wav_path)
                .map_err(|e| format!("{}: {e}", ref_wav_path.display()))?;
            let ref_patches = estimate_patches(ref_wav.len(), cfg);
            if ref_patches > max_patches {
                eprintln!(
                    "truncate {}: ref_wav {} {ref_patches} patches > --max-patches \
                     {max_patches}, using the leading {max_patches} (reference is speaker \
                     conditioning only, never the training target — see the module docs)",
                    row.wav.display(),
                    ref_wav_path.display()
                );
                truncated_refs += 1;
            }
        }

        retained_seconds += wav.len() as f64 / f64::from(REF_RATE);
        kept.push(row);
    }

    eprintln!(
        "manifest filter: {} row(s) kept, {skipped} skipped, {truncated_refs} reference(s) \
         truncated, {retained_seconds:.1}s retained (--max-patches {max_patches})",
        kept.len()
    );
    // MANDATORY, never drop this line. Removing the old hard error on a
    // missing `ref_wav` removed the only thing that caught a manifest whose
    // `ref_wav` column got renamed or lost. Without the printed split such a
    // run trains entirely zero-shot, looks healthy, and surfaces days later
    // as a model that never learned reference cloning.
    let with_ref = kept.iter().filter(|row| row.ref_wav.is_some()).count();
    eprintln!(
        "manifest: {with_ref} row(s) with reference, {} without (the reference VoxCPM \
         implementation recommends 30-50% \
         of rows WITH a reference, so most rows train reference-free; that is what keeps \
         zero-shot cloning alive)",
        kept.len() - with_ref
    );
    if kept.is_empty() {
        return Err(format!(
            "every one of {} manifest row(s) exceeds --max-patches {max_patches}; nothing to \
             train on",
            rows.len()
        )
        .into());
    }
    Ok(kept)
}

/// Read a manifest wav as mono 16 kHz PCM, matching `voxcpm_clone.rs`'s
/// `load_reference` exactly (`decode_audio` plus `to_mono_at_rate`).
pub fn load_wav_16k(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint)?;
    Ok(to_mono_at_rate(&data, REF_RATE)?)
}

/// Decode `row`'s target and (truncated) reference audio, encode both
/// through the AudioVAE, tokenize the text, and run `prefill_capturing` —
/// the exact per-row setup the training loop and the eval-batch builder both
/// need. Shared here so the sequence is defined once. A row without a
/// `ref_wav` builds the zero-shot form: `prefill_capturing` gets `None` and
/// the reference prefix is absent, so `S == text_token_ids.len()`.
pub fn build_prefill_and_target<R, C>(
    model: &VoxCpm2Model<R>,
    client: &C,
    tokenizer: &AnyTokenizer,
    row: &ManifestRow,
    max_patches: usize,
) -> Result<(PrefillState<R>, Tensor<R>), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R> + TypeConversionOps<R> + 'static,
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
        + TypeConversionOps<R>
        + DequantOps<R>,
{
    let wav = load_wav_16k(&row.wav).map_err(|e| format!("{}: {e}", row.wav.display()))?;
    // The training target: what the loss is computed against.
    let target_patches = model.encode_reference(client, &wav)?;

    // The reference-conditioning clip MUST be a different clip than `wav` —
    // see `finetune.rs`'s module docs for why self-referencing is degenerate. A row with
    // no `ref_wav` trains zero-shot (`None`), never a silent fallback to
    // `target_patches`.
    let ref_patches = match &row.ref_wav {
        Some(ref_wav_path) => {
            let ref_wav = load_wav_16k(ref_wav_path)
                .map_err(|e| format!("{}: {e}", ref_wav_path.display()))?;
            // Truncate, never drop: `ref_wav` is speaker conditioning only,
            // not the loss target — see [`truncate_reference`] and the module
            // docs' "Bounding training memory" section.
            let ref_wav = truncate_reference(ref_wav, &model.config, max_patches);
            Some(model.encode_reference(client, &ref_wav)?)
        }
        None => None,
    };

    let normalized = normalize_whitespace(&row.text);
    let mut text_token_ids = tokenize(tokenizer, &normalized);
    // `prefill` requires the sequence to end here: AUDIO_START_ID is the
    // position the first (only, here) patch attends from.
    text_token_ids.push(AUDIO_START_ID);
    // S, exactly: the reference prefix contributes `t_ref + 2` rows, and
    // nothing at all when there is no reference.
    let max_length = match &ref_patches {
        Some(ref_patches) => ref_patches.shape()[0] + 2 + text_token_ids.len(),
        None => text_token_ids.len(),
    };

    // ALWAYS `prefill_capturing`: `cfm_loss`'s teacher-forced path needs
    // `PrefillState::intermediates`, and every row here has a non-empty
    // prefix — see `finetune.rs`'s module docs.
    let prefill =
        model.prefill_capturing(client, ref_patches.as_ref(), &text_token_ids, max_length)?;
    Ok((prefill, target_patches))
}

/// One eval row: the manifest row plus the FIXED `t`/`noise` it is always
/// scored with. Only weight-INDEPENDENT state is cached here. `prefill` is
/// deliberately NOT cached: it runs the base and residual LMs, whose
/// `q_proj`/`v_proj` are the default LoRA targets, so a cached `prefill`
/// would pin the conditioning to the weights at initialization and
/// `eval/diff` would never see the LM learn. See `finetune.rs`'s module docs' "Eval
/// batch" section.
pub struct EvalRow<'a, R: Runtime> {
    pub row: &'a ManifestRow,
    pub t: Tensor<R>,
    pub noise: Tensor<R>,
}

/// Build the fixed eval batch from the LAST `eval_rows` of `kept_rows`,
/// drawing each row's `t`/`noise` ONCE from [`EVAL_NOISE_SEED`] — never
/// re-derived per epoch, never mixed with `args.seed` or `step_counter`, so
/// the eval metric stays comparable across runs and across steps within a
/// run. See `finetune.rs`'s module docs' "Eval batch" section.
pub fn build_eval_batch<'a, R, C>(
    model: &VoxCpm2Model<R>,
    client: &C,
    tokenizer: &AnyTokenizer,
    eval_source_rows: &[&'a ManifestRow],
    max_patches: usize,
) -> Result<Vec<EvalRow<'a, R>>, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static,
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
        + TypeConversionOps<R>
        + DequantOps<R>,
{
    let mut eval_batch = Vec::with_capacity(eval_source_rows.len());
    for (eval_index, row) in eval_source_rows.iter().enumerate() {
        // Built once here ONLY to read the target's shape and dtype, which
        // fix `t`/`noise`. Both are dropped immediately: every eval pass
        // rebuilds them against the CURRENT weights (see `EvalRow`).
        let (prefill, target_patches) =
            build_prefill_and_target(model, client, tokenizer, row, max_patches)?;
        let tcount = target_patches.shape()[0];
        let dtype = target_patches.dtype();
        let noise_shape = target_patches.shape().to_vec();
        drop(prefill);
        drop(target_patches);
        // Stride 2, matching `train_losses`'s own `seed`/`seed + 1` split
        // between the timestep and noise draws — each eval row gets its own
        // pair of streams off the same fixed base, never reused across rows.
        let row_seed = EVAL_NOISE_SEED.wrapping_add((eval_index as u64).wrapping_mul(2));
        let t = client.rand_seeded(&[tcount], dtype, row_seed)?;
        let noise = client.randn_seeded(&noise_shape, dtype, row_seed.wrapping_add(1))?;
        eval_batch.push(EvalRow { row, t, noise });
    }
    Ok(eval_batch)
}

/// Mean `(diff, stop, total)` over `eval_batch`, scored with `drop_cond =
/// false` (the conditioned branch — what inference actually runs; the
/// `finetune.rs` module docs explain why this makes eval not perfectly apples-to-apples
/// with train's CFG-dropout-mixed `loss/diff`). Forward only: never calls
/// `backward` or the optimizer. numr's autograd has no no-grad/detached-
/// forward context (checked: no `no_grad`/`NoGrad` construct exists, only
/// per-`Var` `detach()`/`requires_grad()`, and the model's own LoRA `Var`s
/// still require grad through this call), so each row is built, scored, and
/// dropped before the next row starts, keeping graph retention from stacking
/// across the batch.
///
/// `prefill`/`target_patches` are rebuilt HERE, every call, against the
/// current weights — only `t`/`noise` come cached from [`EvalRow`]. That is
/// what makes `eval/diff` a learning signal: the sampling is pinned, the
/// model is not.
pub fn score_eval_batch<R, C>(
    model: &VoxCpm2Model<R>,
    generator: &PatchGenerator<'_, R>,
    client: &C,
    tokenizer: &AnyTokenizer,
    eval_batch: &[EvalRow<'_, R>],
    max_patches: usize,
    lambda_stop: f64,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R> + TypeConversionOps<R> + 'static,
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
        + TypeConversionOps<R>
        + DequantOps<R>,
{
    let mut diff_sum = 0.0f64;
    let mut stop_sum = 0.0f64;
    let mut total_sum = 0.0f64;
    for eval_row in eval_batch {
        let (prefill, target_patches) =
            build_prefill_and_target(model, client, tokenizer, eval_row.row, max_patches)?;
        let losses = generator.train_losses_with_noise(
            client,
            &prefill,
            &target_patches,
            &eval_row.t,
            &eval_row.noise,
            LAMBDA_DIFF,
            lambda_stop,
            false,
        )?;
        diff_sum += losses.diff.tensor().to_vec::<f32>()[0] as f64;
        stop_sum += losses.stop.tensor().to_vec::<f32>()[0] as f64;
        total_sum += losses.total.tensor().to_vec::<f32>()[0] as f64;
        // `losses` (and the autograd graph it pinned alive) drops here,
        // before the next row's forward pass starts.
    }
    let n = eval_batch.len() as f64;
    Ok((diff_sum / n, stop_sum / n, total_sum / n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// A present `ref_wav` cell resolves to `Some`; an empty cell and a short
    /// row (the column exists in the header but this line has fewer fields)
    /// both resolve to `None` — the switch that decides which rows the
    /// training loop builds a zero-shot prefill for.
    #[test]
    fn load_manifest_ref_wav_column_is_optional_per_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("manifest.tsv");
        let mut file = std::fs::File::create(&path).expect("create manifest");
        writeln!(file, "wav\ttext\tref_wav").expect("write header");
        writeln!(file, "a.wav\thello\tref_a.wav").expect("write row with ref_wav");
        writeln!(file, "b.wav\tworld\t").expect("write row with an empty ref_wav cell");
        writeln!(file, "c.wav\tagain").expect("write a short row with no ref_wav cell at all");
        drop(file);

        let rows = load_manifest(&path).expect("load_manifest");

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].ref_wav, Some(dir.path().join("ref_a.wav")));
        assert_eq!(rows[1].ref_wav, None, "an empty ref_wav cell must be None");
        assert_eq!(
            rows[2].ref_wav, None,
            "a short row must be None, not an error"
        );
    }

    /// `ref_wav` absent from the header ENTIRELY (not just some rows) must
    /// leave every row `None` — the manifest-lost-its-column failure mode
    /// `filter_rows_by_patch_cap`'s printed split exists to catch.
    #[test]
    fn load_manifest_without_a_ref_wav_column_is_all_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("manifest.tsv");
        let mut file = std::fs::File::create(&path).expect("create manifest");
        writeln!(file, "wav\ttext").expect("write header");
        writeln!(file, "a.wav\thello").expect("write row");
        drop(file);

        let rows = load_manifest(&path).expect("load_manifest");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ref_wav, None);
    }
}
