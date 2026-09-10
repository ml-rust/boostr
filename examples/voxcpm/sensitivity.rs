//! Per-tensor task-sensitivity measurement for a VoxCPM2 checkpoint.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_sensitivity -- \
//!     --ckpt CKPT_DIR --audiovae audiovae.safetensors --manifest FILE.tsv \
//!     [--device cpu|cuda] [--encoding Q4AS32D_T64] [--eval-rows 4] \
//!     [--max-patches 38] [--lambda-stop 1.0] [--reference-bpw 16] \
//!     [--baseline-every 32] [--top 20] [--skip 0] [--limit 0]
//! ```
//!
//! # What this answers
//!
//! Which tensors a target encoding may NOT be spent on, ranked by a measured
//! TASK metric rather than by weight-space reconstruction error.
//!
//! TCF's `PROFILES.md` Section 4 solves a profile in six steps. Step 1 is a
//! cheap screen — relative RMS of a dequantized tensor against its source is
//! one — and Section 4 states outright that the screen "never decides final
//! precision". Step 2 is the measurement that does: for each candidate, the
//! task-metric delta from encoding THAT tensor and nothing else. This example
//! is step 2.
//!
//! The distinction is not academic. Relative RMS is the objective TCF's own
//! quantizer minimizes, so ranking by it scores the encoder under its own
//! loss function and predicts nothing about downstream quality. A task metric
//! is a different measurement entirely.
//!
//! # The measurement
//!
//! 1. Load the checkpoint ONCE, dense F32.
//! 2. Score the fixed eval batch. That is the baseline.
//! 3. For each quantizable weight tensor, in turn: snapshot its bytes,
//!    overwrite its values with its OWN quantize -> dequantize round trip at
//!    `--encoding`, re-score the same eval batch, record the delta, then
//!    write the snapshot back. One tensor perturbed at a time; every other
//!    tensor stays at full precision for the whole run.
//! 4. Rank by task delta per byte the encoding would save.
//!
//! The loss is the one `finetune.rs` already scores under `--eval-only`:
//! teacher-forced CFM loss plus stop loss, over a fixed row set, with `t` and
//! the noise drawn once from a fixed seed. It is not re-implemented here —
//! both binaries call the SAME `eval_common::score_eval_batch`.
//!
//! # Why `--ckpt` only
//!
//! `--gguf` and `--tcf` are deliberately not accepted. Those loaders keep a
//! matmul weight PACKED, and a packed weight has no `Var` behind it, so
//! `named_parameters()` does not list it. Perturbing such a model would
//! silently measure a fraction of its tensors and report the rest as absent.
//! Worse, the weights are already quantized, so a round trip on top of them
//! would measure the second encoding, not the first. The measurement needs a
//! full-precision base, which is what a checkpoint directory is.
//!
//! # Encodings
//!
//! `--encoding` takes any `SPECIFICATION.md` Section 12 native identifier:
//! `Q4S32_T64`, `Q4AS32_T64`, `Q4AS32D_T64`, `Q4AS64_T64`, `Q6S32_T64`,
//! `Q6S16D_T64`, `Q8S32_T64`. The default is `Q4AS32D_T64`, `TCF-COMPACT`'s
//! and `TCF-BALANCED`'s base encoding.
//!
//! The ranking is meaningful ONLY relative to the base encoding a profile
//! actually uses: a tensor's delta at `Q4AS32D_T64` says nothing about its
//! delta at `Q8S32_T64`. Two rankings at two encodings are two measurements,
//! never one.
//!
//! Quantization uses `tcf_core`'s real `quantize` and `dequantize_into` —
//! the reference codec, not an approximation of rounding.
//!
//! # Which tensors are candidates
//!
//! `Module::named_parameters()` on the loaded model, filtered to the tensors
//! a native encoding can legally hold (Section 12): rank at least 2, and a
//! trailing dimension that is a whole number of 64-element tiles. Everything
//! rejected is REPORTED with its reason, never silently dropped, so the
//! candidate count is auditable from the output.
//!
//! The AudioVAE is out of scope for the same reason `named_parameters()`
//! excludes it: it is a separately loaded frozen codec, not part of the
//! artifact a profile allocates.
//!
//! Two parameters that share one storage buffer (tied weights) would be
//! perturbed together, which is not a single-tensor measurement. The second
//! and later aliases of a buffer are reported and skipped.
//!
//! # Determinism
//!
//! The comparison is a difference of two losses, so the sampling must be
//! identical across every tensor's measurement or the ranking is noise.
//! Pinned, all of it inherited from `eval_common`:
//!
//! - `t` and the noise: drawn ONCE per eval row from `EVAL_NOISE_SEED`, at
//!   row-indexed stride 2, BEFORE the first perturbation, and cached in the
//!   eval batch for the whole run. Never redrawn, never derived from the
//!   tensor index.
//! - Row membership and row order: the manifest in file order, the
//!   `--max-patches` filter preserving that order, then a suffix by index.
//!   No shuffle, no RNG.
//! - The conditioning branch: `drop_cond = false` always.
//! - The reduction: `diff_sum / n` over rows visited in that fixed order, so
//!   the floating-point summation order is fixed too.
//! - Candidate order: `named_parameters()`'s own order, which is a fixed
//!   traversal of the model, not a hash-map iteration.
//! - Model state at the start of every measurement: the baseline weights,
//!   restored byte for byte after the previous one.
//!
//! Two runs of this example over one checkpoint, manifest, `--eval-rows` and
//! `--encoding`, on one build and one device, produce identical rankings.
//!
//! NOT established as deterministic, and deliberately not claimed: agreement
//! ACROSS devices or ACROSS builds. Different kernels and any thread-order
//! reduction inside one are free to differ. Hold both fixed for a
//! comparison; `finetune.rs`'s module docs make the same reservation for the
//! same reason.
//!
//! # Restoration
//!
//! A silent accumulation of perturbations would invalidate every measurement
//! after the first, so restoration is checked twice, two different ways:
//!
//! - Per tensor. The original bytes are snapshotted before the write and
//!   written back after the score. The buffer is then READ BACK and compared
//!   to the snapshot bit for bit. A mismatch aborts the run.
//! - Periodically. Every `--baseline-every` measurements, and once at the
//!   end, the baseline is re-scored and compared to the first baseline BIT
//!   for bit (`f64::to_bits`, not an epsilon). This catches drift the
//!   per-tensor check cannot see, such as a perturbation reaching a buffer
//!   through an alias.
//!
//! Any drift is a hard error naming the tensor last measured. It is never a
//! warning: a ranking built on drifted weights is worse than no ranking.
//!
//! # Cost
//!
//! The model loads ONCE. Each measurement is one forward pass over the eval
//! rows, plus one quantize/dequantize round trip over the tensor's elements,
//! which is negligible beside the forward pass. So the run is
//! `candidates * eval_rows` forward passes, plus `candidates /
//! baseline_every` more for the drift checks, plus one for the baseline.
//!
//! `--eval-rows` is therefore the cost dial. A screening pass over every
//! candidate uses few rows; a confirmation pass over the top of that ranking
//! uses many. `--skip` and `--limit` split one sweep into shards that can be
//! run separately and concatenated, since every shard measures against the
//! same baseline.
//!
//! Progress is printed per candidate on stderr so a long run is monitorable.
//!
//! # Output
//!
//! One JSON object per line on STDOUT, nothing else — every human-readable
//! line goes to stderr, so two runs diff directly. Records:
//!
//! - `"baseline"`, once, first: the loss the deltas are relative to, plus
//!   the encoding, the row count, the seed and the candidate counts.
//! - `"sensitivity"`, one per measured tensor: name, element count, absolute
//!   and baseline-relative delta, bytes saved, and delta per byte saved.
//! - `"skipped"`, one per rejected parameter, with the reason.
//!
//! The final table on stderr is the same data ranked by delta per byte
//! saved, which is `PROFILES.md` Section 4 step 3's ordering. A NEGATIVE
//! delta means the round trip improved the loss on this eval set; it is
//! reported as measured, never clamped.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use boostr::format::tcf::encoding_name;
use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::{VoxCpmClient, load_tokenizer};
use boostr::nn::Module;
use boostr::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;
use tcf_core::{Encoding, NativeEncoding, dequantize_into, quantize, tile_count};

// The manifest reader, the `--max-patches` filter, the per-row
// prefill/target build, and the eval loss — shared verbatim with
// `finetune.rs`, so this example scores the same number `--eval-only` does.
mod eval_common;
use eval_common::{
    DEFAULT_EVAL_ROWS, DEFAULT_MAX_PATCHES, EVAL_NOISE_SEED, LAMBDA_DIFF, ManifestRow,
    build_eval_batch, filter_rows_by_patch_cap, load_manifest, score_eval_batch,
};

/// `TCF-COMPACT`'s and `TCF-BALANCED`'s base encoding (`PROFILES.md`
/// Section 2), so the default ranking is the one those profiles are solved
/// against.
const DEFAULT_ENCODING: NativeEncoding = NativeEncoding::Q4AS32DT64;
/// Bits per weight the saving is measured AGAINST: the checkpoint's own
/// stored width. A safetensors VoxCPM2 checkpoint is BF16, so a byte saving
/// quoted against 16 bpw is the saving a producer actually realizes.
/// `--reference-bpw` overrides it for a profile whose baseline is another
/// encoding rather than the source checkpoint.
const DEFAULT_REFERENCE_BPW: f64 = 16.0;
/// Measurements between two baseline drift checks. Small enough that a drift
/// is caught near the tensor that caused it, large enough that the checks
/// stay a minor share of the run.
const DEFAULT_BASELINE_EVERY: usize = 32;
/// Rows of the final ranked table printed on stderr. The full ranking is
/// always on stdout regardless.
const DEFAULT_TOP: usize = 20;

/// Runtime to load the model and score on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Device {
    Cpu,
    Cuda,
}

fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => Err(format!(
            "--device: expected one of cpu, cuda, got {other:?}"
        )),
    }
}

/// Parse a `SPECIFICATION.md` Section 12 native encoding identifier. The
/// spelling is `encoding_name`'s exactly, so a name printed by any of these
/// examples is a name this flag accepts.
fn parse_encoding(value: &str) -> Result<NativeEncoding, String> {
    match value {
        "Q4S32_T64" => Ok(NativeEncoding::Q4S32T64),
        "Q4AS32_T64" => Ok(NativeEncoding::Q4AS32T64),
        "Q4AS32D_T64" => Ok(NativeEncoding::Q4AS32DT64),
        "Q4AS64_T64" => Ok(NativeEncoding::Q4AS64T64),
        "Q6S32_T64" => Ok(NativeEncoding::Q6S32T64),
        "Q6S16D_T64" => Ok(NativeEncoding::Q6S16DT64),
        "Q8S32_T64" => Ok(NativeEncoding::Q8S32T64),
        other => Err(format!(
            "--encoding: expected one of Q4S32_T64, Q4AS32_T64, Q4AS32D_T64, Q4AS64_T64, \
             Q6S32_T64, Q6S16D_T64, Q8S32_T64, got {other:?}"
        )),
    }
}

struct Args {
    ckpt: PathBuf,
    audiovae: PathBuf,
    manifest: PathBuf,
    device: Device,
    encoding: NativeEncoding,
    eval_rows: usize,
    max_patches: usize,
    lambda_stop: f64,
    reference_bpw: f64,
    baseline_every: usize,
    top: usize,
    skip: usize,
    limit: usize,
}

const USAGE: &str = "usage: voxcpm_sensitivity --ckpt DIR --audiovae audiovae.safetensors \
--manifest FILE.tsv (header-named TSV: wav, text, optional ref_wav) \
[--device cpu|cuda] \
[--encoding Q4AS32D_T64 (one of Q4S32_T64, Q4AS32_T64, Q4AS32D_T64, Q4AS64_T64, \
Q6S32_T64, Q6S16D_T64, Q8S32_T64)] \
[--eval-rows 4 (rows scored per measurement, taken from the END of the kept \
rows; 0 means every kept row — this is the cost dial)] \
[--max-patches 38 (caps the target wav's patch count; over-cap targets are \
dropped, over-cap ref_wav clips are truncated)] \
[--lambda-stop 1.0] \
[--reference-bpw 16 (the width the byte saving is measured against)] \
[--baseline-every 32 (measurements between two baseline drift checks; 0 \
disables the periodic check, the final one always runs)] \
[--top 20 (rows of the ranked table on stderr; the full ranking is on stdout)] \
[--skip 0] [--limit 0 (0 means no limit; --skip/--limit split one sweep into \
shards that concatenate, since every shard measures against the same baseline)]";

/// Consume the value that follows `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut ckpt = None;
    let mut audiovae = None;
    let mut manifest = None;
    let mut device = Device::Cpu;
    let mut encoding = DEFAULT_ENCODING;
    let mut eval_rows = DEFAULT_EVAL_ROWS;
    let mut max_patches = DEFAULT_MAX_PATCHES;
    let mut lambda_stop = 1.0f64;
    let mut reference_bpw = DEFAULT_REFERENCE_BPW;
    let mut baseline_every = DEFAULT_BASELINE_EVERY;
    let mut top = DEFAULT_TOP;
    let mut skip = 0usize;
    let mut limit = 0usize;

    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        match flag.as_str() {
            "--ckpt" => ckpt = Some(PathBuf::from(take_value(&argv, &mut i, &flag)?)),
            "--audiovae" => audiovae = Some(PathBuf::from(take_value(&argv, &mut i, &flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(take_value(&argv, &mut i, &flag)?)),
            "--device" => device = parse_device(&take_value(&argv, &mut i, &flag)?)?,
            "--encoding" => encoding = parse_encoding(&take_value(&argv, &mut i, &flag)?)?,
            "--eval-rows" => {
                eval_rows = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--eval-rows: {e}"))?;
            }
            "--max-patches" => {
                max_patches = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--max-patches: {e}"))?;
            }
            "--lambda-stop" => {
                lambda_stop = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--lambda-stop: {e}"))?;
            }
            "--reference-bpw" => {
                reference_bpw = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--reference-bpw: {e}"))?;
            }
            "--baseline-every" => {
                baseline_every = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--baseline-every: {e}"))?;
            }
            "--top" => {
                top = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--top: {e}"))?;
            }
            "--skip" => {
                skip = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--skip: {e}"))?;
            }
            "--limit" => {
                limit = take_value(&argv, &mut i, &flag)?
                    .parse()
                    .map_err(|e| format!("--limit: {e}"))?;
            }
            // Named explicitly rather than falling into "unknown flag": a
            // quantized artifact is the one input that looks like it should
            // work here and cannot — see the module docs.
            "--gguf" | "--tcf" => {
                return Err(format!(
                    "{flag}: this measurement needs a full-precision base, so only --ckpt is \
                     accepted. A packed weight has no Var behind it and would not be \
                     perturbed at all, and a weight already quantized would measure the \
                     second encoding rather than the first.\n{USAGE}"
                ));
            }
            other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
        }
        i += 1;
    }

    if max_patches == 0 {
        return Err(format!("--max-patches must be at least 1\n{USAGE}"));
    }
    if !(reference_bpw.is_finite() && reference_bpw > 0.0) {
        return Err(format!(
            "--reference-bpw must be a positive number\n{USAGE}"
        ));
    }
    Ok(Args {
        ckpt: ckpt.ok_or_else(|| format!("--ckpt is required\n{USAGE}"))?,
        audiovae: audiovae.ok_or_else(|| format!("--audiovae is required\n{USAGE}"))?,
        manifest: manifest.ok_or_else(|| format!("--manifest is required\n{USAGE}"))?,
        device,
        encoding,
        eval_rows,
        max_patches,
        lambda_stop,
        reference_bpw,
        baseline_every,
        top,
        skip,
        limit,
    })
}

/// One measured tensor.
struct Measurement {
    name: String,
    elements: usize,
    /// `perturbed_total - baseline_total`, in the loss's own units.
    delta: f64,
    /// `delta / |baseline_total|`. The number that compares across
    /// checkpoints, where the absolute loss scale differs.
    relative_delta: f64,
    /// Bytes this encoding saves on this tensor against `--reference-bpw`.
    bytes_saved: f64,
    /// `delta / bytes_saved` — `PROFILES.md` Section 4 step 3's ordering.
    delta_per_byte: f64,
}

/// A parameter that could not be measured, and why. Reported rather than
/// dropped: a candidate count nobody can audit is a candidate count nobody
/// can trust.
struct Skipped {
    name: String,
    elements: usize,
    reason: String,
}

/// Overwrite `tensor`'s device buffer with `values`.
///
/// The buffer is addressed through `Tensor::ptr`, which already carries the
/// layout offset, and written with `Runtime::copy_to_device` — the same safe
/// entry point boostr's own sampling and GGUF-reader paths use to fill a
/// tensor in place. Only ever called on a contiguous F32 tensor, checked by
/// the caller, so the byte count is exactly `values.len() * 4`.
fn write_values<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: &[u8] = bytemuck::cast_slice(values);
    R::copy_to_device(bytes, tensor.ptr(), tensor.device())?;
    Ok(())
}

/// Whether a native encoding can legally hold a tensor of this shape:
/// `SPECIFICATION.md` Section 12 requires rank at least 2 and a trailing
/// dimension that is a whole number of tiles. Asked by calling the tiling
/// unit itself rather than by re-deriving the rule here, so the answer can
/// never disagree with the encoder's.
fn tiling_error(dims: &[u64], tile: u16) -> Option<String> {
    match tile_count(dims, dims.len() as u32, tile) {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    }
}

/// Load the model, score the baseline, then perturb one tensor at a time.
fn run<R: Runtime<DType = DType>>(
    args: &Args,
    device: &R::Device,
    client: &(impl VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static),
    rows: &[ManifestRow],
    started: Instant,
) -> Result<(), Box<dyn std::error::Error>>
where
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
    // Dense F32, matching `finetune.rs`'s own `--ckpt` arm. The perturbation
    // reads and writes f32 directly, so a cast here would add a rounding the
    // measurement does not intend.
    eprintln!("loading {} ...", args.ckpt.display());
    let model =
        VoxCpm2Model::<R>::from_checkpoint(&args.ckpt, &args.audiovae, device, Some(DType::F32))?;

    let rows = filter_rows_by_patch_cap(rows, &model.config, args.max_patches)?;
    // The whole manifest is the held-out set here, exactly as it is under
    // `--eval-only`: there is no training half to protect. A nonzero
    // `--eval-rows N` takes the LAST N kept rows by index, so this example
    // and a `finetune --eval-only` run over one manifest score the same rows
    // in the same order.
    let eval_source_rows: Vec<&ManifestRow> = if args.eval_rows == 0 || args.eval_rows >= rows.len()
    {
        rows.clone()
    } else {
        rows[rows.len() - args.eval_rows..].to_vec()
    };
    eprintln!("eval batch: {} row(s)", eval_source_rows.len());

    let tokenizer = load_tokenizer(args.ckpt.join("tokenizer.json"))?;
    // Drawn ONCE, before the first perturbation, and reused for every
    // measurement — see the module docs' "Determinism" section.
    let eval_batch = build_eval_batch(
        &model,
        client,
        &tokenizer,
        &eval_source_rows,
        args.max_patches,
    )?;

    let score = |label: &str| -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
        let generator = model.patch_generator();
        score_eval_batch(
            &model,
            &generator,
            client,
            &tokenizer,
            &eval_batch,
            args.max_patches,
            args.lambda_stop,
        )
        .map_err(|e| format!("{label}: {e}").into())
    };

    let (base_diff, base_stop, base_total) = score("baseline")?;
    eprintln!(
        "baseline: eval/diff {base_diff:.6} eval/stop {base_stop:.6} eval/total {base_total:.6}"
    );

    let layout = args.encoding.layout();
    let encoded_bpw = layout.bits_per_weight();
    let name = encoding_name(Encoding::Native(args.encoding));

    // Owned handles, taken before the loop so no `Var` borrow is held across
    // a scoring call. A cloned `Tensor` shares its storage, so writing
    // through it writes the buffer the model reads.
    let mut candidates: Vec<(String, Tensor<R>)> = Module::named_parameters(&model)
        .into_iter()
        .map(|(name, var)| (name, var.tensor().clone()))
        .collect();
    let parameter_count = candidates.len();

    let mut skipped: Vec<Skipped> = Vec::new();
    let mut seen_buffers: HashSet<u64> = HashSet::new();
    candidates.retain(|(name, tensor)| {
        let elements = tensor.numel();
        let mut reject = |reason: String| {
            skipped.push(Skipped {
                name: name.clone(),
                elements,
                reason,
            });
            false
        };
        if tensor.dtype() != DType::F32 {
            return reject(format!("dtype {:?}, expected F32", tensor.dtype()));
        }
        if !tensor.is_contiguous() {
            return reject("not contiguous: its buffer cannot be rewritten in place".to_string());
        }
        let dims: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
        if let Some(reason) = tiling_error(&dims, layout.geometry.tile) {
            return reject(format!("no native encoding can hold this shape: {reason}"));
        }
        // A tied parameter shares one buffer with an earlier one. Perturbing
        // it would move two tensors at once, which is not the measurement.
        if !seen_buffers.insert(tensor.ptr()) {
            return reject("shares its buffer with an earlier parameter (tied weight)".to_string());
        }
        true
    });

    let total_candidates = candidates.len();
    let selected: Vec<(String, Tensor<R>)> = candidates
        .into_iter()
        .skip(args.skip)
        .take(if args.limit == 0 {
            usize::MAX
        } else {
            args.limit
        })
        .collect();

    eprintln!(
        "candidates: {} of {parameter_count} parameter(s) quantizable at {name}, {} skipped, \
         {} selected by --skip/--limit",
        total_candidates,
        skipped.len(),
        selected.len()
    );

    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "record": "baseline",
            "ckpt": args.ckpt.display().to_string(),
            "manifest": args.manifest.display().to_string(),
            "device": match args.device { Device::Cpu => "cpu", Device::Cuda => "cuda" },
            "encoding": name,
            "encoded_bpw": encoded_bpw,
            "reference_bpw": args.reference_bpw,
            "eval_rows": eval_batch.len(),
            "eval_seed": EVAL_NOISE_SEED,
            "max_patches": args.max_patches,
            "lambda_diff": LAMBDA_DIFF,
            "lambda_stop": args.lambda_stop,
            "parameters": parameter_count,
            "candidates": total_candidates,
            "measured": selected.len(),
            "baseline_diff": base_diff,
            "baseline_stop": base_stop,
            "baseline_total": base_total,
        }))?
    );
    for entry in &skipped {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "record": "skipped",
                "name": entry.name,
                "elements": entry.elements,
                "reason": entry.reason,
            }))?
        );
    }

    let mut measurements: Vec<Measurement> = Vec::with_capacity(selected.len());
    for (index, (tensor_name, tensor)) in selected.iter().enumerate() {
        let elements = tensor.numel();
        let dims: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
        let rank = dims.len() as u32;

        // The snapshot restoration is checked against. Read as f32 rather
        // than as opaque bytes only because the quantizer needs the same
        // values anyway; the restore writes these exact bits back.
        let original: Vec<f32> = tensor.try_to_vec::<f32>()?;

        let tiles = match quantize(&original, &dims, rank, layout) {
            Ok(tiles) => tiles,
            // A tensor the codec refuses (a non-finite value, a scale that
            // underflows binary16) is a result about the tensor, not a fault
            // in the run.
            Err(e) => {
                eprintln!(
                    "[{}/{}] {tensor_name}: not encodable: {e}",
                    index + 1,
                    selected.len()
                );
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "record": "skipped",
                        "name": tensor_name,
                        "elements": elements,
                        "reason": format!("{name} rejected this tensor: {e}"),
                    }))?
                );
                continue;
            }
        };
        let mut round_trip: Vec<f32> = Vec::new();
        dequantize_into(&tiles, layout, &mut round_trip)?;
        drop(tiles);
        if round_trip.len() != original.len() {
            return Err(format!(
                "{tensor_name}: round trip produced {} value(s) for {} element(s)",
                round_trip.len(),
                original.len()
            )
            .into());
        }

        write_values(tensor, &round_trip)?;
        let scored = score(tensor_name);

        // Restored BEFORE the score is unwrapped, so a scoring error still
        // leaves the model at its baseline rather than perturbed.
        write_values(tensor, &original)?;
        let restored: Vec<f32> = tensor.try_to_vec::<f32>()?;
        if restored.len() != original.len()
            || restored
                .iter()
                .zip(&original)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(format!(
                "{tensor_name}: restoration did not reproduce the original bytes; every \
                 measurement after this one would be scored against drifted weights"
            )
            .into());
        }

        let (_, _, perturbed_total) = scored?;
        let delta = perturbed_total - base_total;
        let relative_delta = if base_total == 0.0 {
            delta
        } else {
            delta / base_total.abs()
        };
        let bytes_saved = elements as f64 * (args.reference_bpw - encoded_bpw) / 8.0;
        let delta_per_byte = if bytes_saved == 0.0 {
            f64::NAN
        } else {
            delta / bytes_saved
        };

        eprintln!(
            "[{}/{}] {tensor_name} elems={elements} delta={delta:+.6e} rel={relative_delta:+.6e} \
             bytes_saved={bytes_saved:.0} delta/byte={delta_per_byte:+.6e}",
            index + 1,
            selected.len()
        );
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "record": "sensitivity",
                "name": tensor_name,
                "elements": elements,
                "encoding": name,
                "delta": delta,
                "relative_delta": relative_delta,
                "bytes_saved": bytes_saved,
                "delta_per_byte": delta_per_byte,
                "perturbed_total": perturbed_total,
            }))?
        );
        measurements.push(Measurement {
            name: tensor_name.clone(),
            elements,
            delta,
            relative_delta,
            bytes_saved,
            delta_per_byte,
        });

        let due = args.baseline_every > 0 && (index + 1) % args.baseline_every == 0;
        if due {
            check_baseline_drift(&score, base_total, tensor_name)?;
            eprintln!(
                "baseline re-check after {} measurement(s): unchanged",
                index + 1
            );
        }
    }

    // Always, regardless of `--baseline-every`: a run whose last measurement
    // is unverified has an unverified ranking.
    if let Some(last) = measurements.last() {
        check_baseline_drift(&score, base_total, &last.name)?;
        eprintln!("final baseline re-check: unchanged");
    }

    report(&measurements, args.top, &name);
    eprintln!("total {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

/// Re-score the baseline and compare it BIT for bit to the first one.
///
/// An epsilon comparison would hide exactly the drift this exists to catch:
/// a partially restored buffer moves the loss by a tiny amount, and a
/// tolerance large enough to absorb it is large enough to absorb a real
/// sensitivity delta too. The two scores run identical arithmetic over
/// identical inputs, so equality is the correct test.
fn check_baseline_drift<F>(
    score: &F,
    base_total: f64,
    after: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&str) -> Result<(f64, f64, f64), Box<dyn std::error::Error>>,
{
    let (_, _, again) = score("baseline re-check")?;
    if again.to_bits() != base_total.to_bits() {
        return Err(format!(
            "baseline drifted after measuring {after}: {base_total:.9e} became {again:.9e}. \
             The model no longer holds its original weights, so every measurement in this run \
             is invalid."
        )
        .into());
    }
    Ok(())
}

/// The ranked table on stderr. `PROFILES.md` Section 4 step 3 ranks by task
/// delta per unit of cost, so that is the sort key here; the full ranking is
/// on stdout either way.
fn report(measurements: &[Measurement], top: usize, encoding: &str) {
    if measurements.is_empty() {
        eprintln!("no tensor was measured");
        return;
    }
    let mut ranked: Vec<&Measurement> = measurements.iter().collect();
    ranked.sort_by(|a, b| {
        b.delta_per_byte
            .partial_cmp(&a.delta_per_byte)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let shown = ranked.len().min(top.max(1));
    eprintln!(
        "\ntop {shown} of {} tensor(s) by task delta per byte saved at {encoding} \
         (most sensitive first):",
        ranked.len()
    );
    eprintln!(
        "{:>4}  {:<52} {:>12} {:>13} {:>13} {:>13} {:>13}",
        "rank", "tensor", "elements", "delta", "rel delta", "bytes saved", "delta/byte"
    );
    for (rank, m) in ranked.iter().take(shown).enumerate() {
        eprintln!(
            "{:>4}  {:<52} {:>12} {:>13.5e} {:>13.5e} {:>13.0} {:>13.5e}",
            rank + 1,
            m.name,
            m.elements,
            m.delta,
            m.relative_delta,
            m.bytes_saved,
            m.delta_per_byte
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let started = Instant::now();

    let rows = load_manifest(&args.manifest)?;

    match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime>(&args, &device, &client, &rows, started)?;
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run::<CudaRuntime>(&args, &device, &client, &rows, started)?;
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            eprintln!(
                "--device cuda: this binary was built without CUDA support; rebuild with \
                 --features cuda"
            );
            std::process::exit(2);
        }
    }

    Ok(())
}
