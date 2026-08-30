//! End-to-end VoxCPM2 voice cloning: reference wav plus target text in, a
//! 48 kHz wav out. This is the Rust replacement for
//! `audio/pipeline/voxcpm_clone.py`.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_clone -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) \
//!     [--config config.json] \
//!     --audiovae audiovae.pth \
//!     --ref REF.wav (--text "..." --out OUT.wav \
//!                    | --prompts PROMPTS.tsv --out-dir DIR) [--jsonl LOG.jsonl] \
//!     [--n-timesteps 10] [--cfg 2.0] [--min-len 2] [--max-len N] \
//!     [--seed 0] [--best-of N] [--dtype f32] [--device cpu]
//! ```
//!
//! `CKPT_DIR` holds `config.json`, `model.safetensors` and `tokenizer.json`.
//! `--audiovae` is the separately shipped `audiovae.pth`; an
//! `audiovae.safetensors` converted from it is accepted too, and which one a
//! path holds is read off the file's bytes, not its name.
//!
//! `--gguf` is the single-file alternative, written by `compressr convert
//! CKPT_DIR --format gguf --quantization q4_k`. It is mutually exclusive with
//! `--ckpt`. A GGUF carries the transformer stack ONLY, so `--audiovae` is
//! still required, `--config` supplies the `config.json` the file has no
//! embedded copy of, and `tokenizer.json` is looked for beside the `.gguf`
//! and then beside `--config`.
//!
//! `--tcf` is the third single-file form, written by `compressr convert
//! CKPT_DIR --format tcf`, and is mutually exclusive with both of the above.
//! It carries the transformer stack only, on the same terms as `--gguf`,
//! except that `--config` is REQUIRED rather than optional: the format has no
//! metadata map a `config.json` could ever be embedded in. Every tensor at a
//! native encoding stays PACKED in memory, so a 1.2 GB file costs about 1.2
//! GB rather than its f32 expansion.
//!
//! # Reference mode, never continuation
//!
//! The Python script defaults to `reference_wav_path`, not `prompt_wav_path`,
//! and this port implements only that mode. Continuation concatenates the
//! reference transcript with the target text and asks the model to carry on
//! speaking, which measured as 17 invented words in 14.2 s of audio for a
//! sentence that needs 4.8 s. Reference mode isolates the voice through the
//! reference audio tokens with no continuation semantics, and boostr's
//! `decode_patches` is reference-mode only for the same reason (there is no
//! context prefix to trim).
//!
//! # Why the checks here are structural, not numerical
//!
//! Every numerical stage already has its own gate against the reference
//! implementation: `voxcpm_vae_check`, `voxcpm_locenc_check`,
//! `voxcpm_baselm_check`, `voxcpm_residual_lm_check`, `voxcpm_dit_check`,
//! `voxcpm_cfm_check`, `voxcpm_fsq_check`, `voxcpm_prefill_check`,
//! `voxcpm_step_check`, `voxcpm_tokenizer_check`. An end-to-end tensor
//! comparison is not possible on top of those: the loop is autoregressive and
//! the CFM sampler draws its own noise, so torch and numr cannot be seeded
//! into agreement and drift compounds over hundreds of patches. See
//! `voxcpm_step_check`'s module doc for the measured drift.
//!
//! So this file asserts what an end-to-end run CAN assert, and fails loudly
//! on any of it:
//!
//! 1. The outcome is `StopToken` or `MaxLen`, and a `StopToken` run emitted
//!    more than `min_len + 1` patches (the stop guard is `i > min_len`).
//! 2. `patches * patch_size * HOP_LENGTH == decoded sample count`, with
//!    `HOP_LENGTH` read from the VAE decoder rather than hardcoded.
//! 3. No NaN and no infinity in the decoded waveform.
//! 4. Peak absolute sample <= 1.0.
//! 5. Peak absolute sample > 1e-4, so a model that emits zeros fails here
//!    instead of writing a silent file that looks fine.
//!
//! Every report line goes to stderr, so stdout stays clean.
//!
//! # `--best-of N`
//!
//! Generates N takes with seeds `seed, seed+1, ...` and keeps the one whose
//! median F0 is closest to the reference audio's median F0. Pitch varies a
//! lot between draws: the Python script measured six generations of one
//! sentence at 126-147 Hz against a 137 Hz reference, and a seventh at 176.
//! F0 comes from `boostr::model::audio::estimate_pitch` (YIN), the crate's
//! own estimator, never a hand-rolled one. A take with no measurable pitch is
//! unusable rather than merely a poor match, so it never wins.
//!
//! Each take re-runs `prefill`, because `GenerateState::start` consumes the
//! `PrefillState` and the loop mutates its KV caches in place. The reference
//! encode is done once and shared.
//!
//! # Sweep mode: `--prompts FILE.tsv --out-dir DIR [--jsonl FILE]`
//!
//! ```text
//! cargo run --release --features audio,f16,cuda --example voxcpm_clone -- \
//!     --tcf MODEL.tcf --config config.json --audiovae audiovae.pth \
//!     --ref REF.wav --prompts heldout_prompts.tsv --out-dir renders/ \
//!     --jsonl renders/results.jsonl --device cuda
//! ```
//!
//! `--prompts` is mutually exclusive with `--text`, and takes a tab-separated
//! file with a header row. `id` and `prompt` are REQUIRED columns; `axis`,
//! `value`, `lang` and `register` are carried into the JSONL when present so
//! the matrix can be sliced by prompt class. Columns are addressed by header
//! name, so extra ones appearing later are ignored rather than misread.
//!
//! The point of the flag is that the model, the AudioVAE, the LoRA adapter,
//! the tokenizer and the reference encode are done ONCE and reused for every
//! prompt. Loading a q6/q8 TCF costs 26-27 s, dominated by Section 15 digest
//! verification over 1.9-2.4 GB, so 66 prompts re-invoked per render would
//! spend half an hour purely loading and swamp the numbers being measured.
//!
//! One wav per prompt lands in `--out-dir` named `{id}.wav`.
//!
//! # `--jsonl FILE`
//!
//! One JSON object per line, flushed as each render completes, so a sweep that
//! dies at prompt 300 keeps the first 299. The first line is the run record
//! (`"record":"run"`) carrying the weight source, device, dtype, sampler
//! settings and `load_seconds`; every later line is a render record
//! (`"record":"render"`).
//!
//! `--jsonl` works with `--text` too: the useful numbers are not batch-only.
//!
//! ## Per-render timing is split by phase
//!
//! A render is prefill, then the autoregressive step loop, then the AudioVAE
//! decode, then the wav write, and only the second of those answers to the
//! quantization tier. The tier changes the weights `tcf_gemm_f32` multiplies
//! against; the AudioVAE is loaded from `.pth` at full precision in every row
//! of a tier comparison, and prefill scales with the reference and the prompt.
//! A combined number therefore understates the gap between two tiers by
//! whatever fraction the constant phases happen to be — measured at 42% of an
//! utterance on CPU.
//!
//! So each render reports `prefill_seconds`, `generate_seconds`,
//! `vocode_seconds`, `pitch_seconds` and `write_seconds` alongside the
//! end-to-end `wall_seconds`, plus `generate_rtf` alongside `rtf`.
//! `generate_rtf` is the tier-sensitive number; `rtf` is the honest
//! end-to-end one. Every phase is summed across takes, so `--best-of 4`
//! reports four prefills, four loops and four decodes.
//!
//! `prefill_seconds + generate_seconds + vocode_seconds + pitch_seconds`
//! accounts for `wall_seconds` to within the loop's own bookkeeping.
//! `write_seconds` is measured but sits OUTSIDE `wall_seconds`: the write
//! happens after the checks have decided whether to write at all, and a wall
//! clock whose meaning changed with the outcome would be worse than one that
//! excludes a constant.
//!
//! ## `--best-of N` and wall clock
//!
//! Best-of-N renders N takes and keeps one, so `wall_seconds` covers ALL N
//! takes while `audio_seconds` is the winner's alone, and `rtf` is therefore
//! the cost of the whole selection. The record states `best_of` and `take`
//! (which take won, 1-based) so a best-of-4 row can never be read as one
//! render. Reporting the selection cost as if it were a single render would
//! silently corrupt a comparison matrix; refusing the combination would cost
//! the measurement instead, so both numbers are reported.
//!
//! # Failure handling differs between the two modes
//!
//! A failed structural check in `--text` mode exits 1, unchanged. In sweep
//! mode it records `"checks_passed":false`, writes the wav anyway so the
//! failure can be listened to, continues with the next prompt, and exits 1 at
//! the end. A sweep must not lose 400 good renders to one bad one.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use boostr::format::SafeTensors;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateOutcome, GenerateState, StepOutcome, VoxCpm2Model,
};
use boostr::model::audio::voxcpm::vae::decoder::{HOP_LENGTH, SAMPLE_RATE};
use boostr::model::audio::voxcpm::{VoxCpmClient, load_tokenizer, normalize_whitespace, tokenize};
use boostr::model::audio::{
    PitchOptions, decode_audio, encode_wav_pcm16, estimate_pitch, extension_hint, to_mono_at_rate,
};
use boostr::nn::{LoraTargets, check_lora_metadata};
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

/// Rate the reference wav is resampled to before the AudioVAE encoder. Fixed
/// by the encoder, not a choice: `AudioVaeEncoder` hops 640 samples at
/// 16 kHz.
const REF_RATE: u32 = 16_000;

/// Ceiling on the computed `--max-len`, mirroring the reference pipeline.
const MAX_LEN_CAP: usize = 4096;

/// Patches per progress line. One line per patch would be hundreds of lines
/// for a normal sentence.
const PROGRESS_EVERY: usize = 25;

/// Peak below this counts as silence, not audio.
const SILENCE_EPSILON: f32 = 1e-4;

/// Default LoRA rank, matching `voxcpm_finetune`'s default.
const DEFAULT_LORA_RANK: usize = 16;
/// Default LoRA alpha, matching `voxcpm_finetune`'s default.
const DEFAULT_LORA_ALPHA: f32 = 32.0;
/// Default LoRA target projections, matching `voxcpm_finetune`'s default.
const DEFAULT_LORA_TARGETS: &str = "q_proj,v_proj";

/// Where the transformer stack's weights come from.
///
/// `--ckpt` names a checkpoint DIRECTORY (`config.json`,
/// `model.safetensors`, `tokenizer.json`); `--gguf` and `--tcf` each name a
/// single file that carries the weights and nothing else. Mutually
/// exclusive, and exactly one of them is required.
enum Weights {
    Checkpoint(PathBuf),
    Gguf(PathBuf),
    Tcf(PathBuf),
}

/// Runtime to build the model and run generation on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Device {
    Cpu,
    Cuda,
}

/// Parse a `--device` value.
fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => Err(format!(
            "--device: expected one of cpu, cuda, got {other:?}"
        )),
    }
}

/// Parsed command line.
struct Args {
    weights: Weights,
    /// `config.json` for the single-file paths. Ignored for `--ckpt`, which
    /// reads the one in the checkpoint directory. Optional for `--gguf`,
    /// required for `--tcf`.
    config: Option<PathBuf>,
    audiovae: PathBuf,
    reference: PathBuf,
    /// Single-render target text. Exactly one of `text` and `prompts` is set.
    text: Option<String>,
    /// Output wav for `--text`. Required with it, rejected with `--prompts`.
    out: Option<PathBuf>,
    /// Sweep input: a TSV of held-out prompts, one render each.
    prompts: Option<PathBuf>,
    /// Output directory for `--prompts`, one `{id}.wav` per row.
    out_dir: Option<PathBuf>,
    /// Machine-readable render log, one JSON object per line, flushed per
    /// render. Valid in both modes.
    jsonl: Option<PathBuf>,
    n_timesteps: usize,
    cfg: f32,
    min_len: usize,
    /// `None` until the tokenized text length is known; see [`default_max_len`].
    max_len: Option<usize>,
    seed: u64,
    best_of: usize,
    /// Transformer-stack dtype. `None` keeps every weight at the dtype it has
    /// in the checkpoint (BF16 for VoxCPM2). The `AudioVAE` is never cast.
    dtype: Option<DType>,
    /// Runtime to build the model and run generation on.
    device: Device,
    /// LoRA adapter safetensors file, saved by `voxcpm_finetune`. `None`
    /// runs the base model unchanged.
    lora: Option<PathBuf>,
    /// Rank the adapter was trained at. Must match the adapter file's
    /// `__metadata__` or loading it fails.
    lora_rank: usize,
    /// Alpha the adapter was trained at. Must match the adapter file's
    /// `__metadata__` or loading it fails.
    lora_alpha: f32,
    /// Comma-separated projection names the adapter targets, e.g.
    /// `q_proj,v_proj`. Must match the adapter file's `__metadata__` or
    /// loading it fails.
    lora_targets: String,
}

/// Parse a `--dtype` value into the cast the loader takes.
///
/// `native` means "no cast", which loads the checkpoint's own BF16 weights and
/// roughly halves resident memory against `f32`.
fn parse_dtype(value: &str) -> Result<Option<DType>, String> {
    match value {
        "f32" => Ok(Some(DType::F32)),
        "bf16" => Ok(Some(DType::BF16)),
        "f16" => Ok(Some(DType::F16)),
        "native" => Ok(None),
        other => Err(format!(
            "--dtype: expected one of f32, bf16, f16, native, got {other:?}"
        )),
    }
}

const USAGE: &str = "usage: voxcpm_clone (--ckpt DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) \
[--config config.json] \
--audiovae PATH --ref REF.wav \
(--text \"...\" --out OUT.wav | --prompts FILE.tsv --out-dir DIR) [--jsonl FILE] \
[--n-timesteps 10] [--cfg 2.0] [--min-len 2] \
[--max-len N] [--seed 0] [--best-of 1] [--dtype f32|bf16|f16|native] \
[--device cpu|cuda] [--lora ADAPTER.safetensors] [--lora-rank 16] \
[--lora-alpha 32.0] [--lora-targets q_proj,v_proj]";

/// Hard cap on emitted patches when `--max-len` is not given.
///
/// `text_len` is the tokenized length WITHOUT the appended
/// [`AUDIO_START_ID`], matching the reference pipeline's own arithmetic.
fn default_max_len(text_len: usize) -> usize {
    (text_len * 6 + 10).min(MAX_LEN_CAP)
}

/// Consume the value that follows `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut ckpt, mut audiovae, mut reference, mut text, mut out) = (None, None, None, None, None);
    let mut gguf: Option<PathBuf> = None;
    let mut tcf: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut jsonl: Option<PathBuf> = None;
    let mut n_timesteps = 10usize;
    let mut cfg = 2.0f32;
    let mut min_len = 2usize;
    let mut max_len = None;
    let mut seed = 0u64;
    let mut best_of = 1usize;
    let mut dtype = Some(DType::F32);
    let mut device = Device::Cpu;
    let mut lora: Option<PathBuf> = None;
    let mut lora_rank = DEFAULT_LORA_RANK;
    let mut lora_alpha = DEFAULT_LORA_ALPHA;
    let mut lora_targets = DEFAULT_LORA_TARGETS.to_string();

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--ckpt" => ckpt = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--gguf" => gguf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--tcf" => tcf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--config" => config = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audiovae" => audiovae = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--ref" => reference = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--text" => text = Some(take_value(&argv, &mut i, flag)?),
            "--out" => out = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--prompts" => prompts = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--out-dir" => out_dir = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--jsonl" => jsonl = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--n-timesteps" => {
                n_timesteps = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--n-timesteps: {e}"))?
            }
            "--cfg" => {
                cfg = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--cfg: {e}"))?
            }
            "--min-len" => {
                min_len = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--min-len: {e}"))?
            }
            "--max-len" => {
                max_len = Some(
                    take_value(&argv, &mut i, flag)?
                        .parse()
                        .map_err(|e| format!("--max-len: {e}"))?,
                )
            }
            "--seed" => {
                seed = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--best-of" => {
                best_of = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--best-of: {e}"))?
            }
            "--dtype" => dtype = parse_dtype(&take_value(&argv, &mut i, flag)?)?,
            "--device" => device = parse_device(&take_value(&argv, &mut i, flag)?)?,
            "--lora" => lora = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--lora-rank" => {
                lora_rank = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--lora-rank: {e}"))?
            }
            "--lora-alpha" => {
                lora_alpha = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--lora-alpha: {e}"))?
            }
            "--lora-targets" => lora_targets = take_value(&argv, &mut i, flag)?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }

    if n_timesteps == 0 {
        return Err("--n-timesteps must be at least 1".to_string());
    }
    if best_of == 0 {
        return Err("--best-of must be at least 1".to_string());
    }
    if max_len == Some(0) {
        return Err("--max-len must be at least 1".to_string());
    }
    if lora.is_some() && lora_rank == 0 {
        return Err("--lora-rank must be at least 1".to_string());
    }
    if lora.is_some() && lora_targets.trim().is_empty() {
        return Err("--lora-targets must name at least one projection".to_string());
    }

    // Exactly one weight source. Accepting two and silently preferring one
    // would load a different model than the operator asked for.
    let weights = match (ckpt, gguf, tcf) {
        (Some(dir), None, None) => Weights::Checkpoint(dir),
        (None, Some(path), None) => Weights::Gguf(path),
        (None, None, Some(path)) => Weights::Tcf(path),
        (None, None, None) => {
            return Err(format!("--ckpt, --gguf or --tcf is required\n{USAGE}"));
        }
        _ => {
            return Err(format!(
                "--ckpt, --gguf and --tcf are mutually exclusive\n{USAGE}"
            ));
        }
    };
    // A TCF has no metadata map to embed a config.json in, so the path is
    // the only way the architecture can be known. Caught here rather than
    // after the 1.2 GB file has been mapped and verified.
    if matches!(weights, Weights::Tcf(_)) && config.is_none() {
        return Err(format!("--config is required with --tcf\n{USAGE}"));
    }

    // One render or a sweep, never both, and each takes its own output flag.
    // Accepting `--text` with `--out-dir` would silently ignore one of them.
    match (&text, &prompts) {
        (Some(_), Some(_)) => {
            return Err(format!(
                "--text and --prompts are mutually exclusive\n{USAGE}"
            ));
        }
        (None, None) => return Err(format!("--text or --prompts is required\n{USAGE}")),
        (Some(_), None) => {
            if out.is_none() {
                return Err(format!("--out is required with --text\n{USAGE}"));
            }
            if out_dir.is_some() {
                return Err(format!(
                    "--out-dir belongs to --prompts, not --text\n{USAGE}"
                ));
            }
        }
        (None, Some(_)) => {
            if out_dir.is_none() {
                return Err(format!("--out-dir is required with --prompts\n{USAGE}"));
            }
            if out.is_some() {
                return Err(format!("--out belongs to --text, not --prompts\n{USAGE}"));
            }
        }
    }

    Ok(Args {
        weights,
        config,
        audiovae: audiovae.ok_or_else(|| format!("--audiovae is required\n{USAGE}"))?,
        reference: reference.ok_or_else(|| format!("--ref is required\n{USAGE}"))?,
        text,
        out,
        prompts,
        out_dir,
        jsonl,
        n_timesteps,
        cfg,
        min_len,
        max_len,
        seed,
        best_of,
        dtype,
        device,
        lora,
        lora_rank,
        lora_alpha,
        lora_targets,
    })
}

/// Locate `tokenizer.json`.
///
/// A checkpoint directory holds it outright. Neither a GGUF nor a TCF carries
/// a tokenizer at all, so it is looked for beside the model file first and
/// beside `--config` second — both of those normally sit in, or are copied from, the same
/// checkpoint directory. Neither: an error, rather than a tokenizer guess
/// that would silently produce the wrong token ids.
fn tokenizer_path(weights: &Weights, config: Option<&Path>) -> Result<PathBuf, String> {
    match weights {
        Weights::Checkpoint(dir) => Ok(dir.join("tokenizer.json")),
        Weights::Gguf(path) | Weights::Tcf(path) => {
            let beside = |p: &Path| {
                p.parent()
                    .map(|dir| dir.join("tokenizer.json"))
                    .filter(|candidate| candidate.is_file())
            };
            beside(path)
                .or_else(|| config.and_then(beside))
                .ok_or_else(|| {
                    format!(
                        "no tokenizer.json beside {} (a single-file model carries none); \
                     put it there or pass --config pointing into the checkpoint \
                     directory",
                        path.display()
                    )
                })
        }
    }
}

/// One render the run will perform: `--text` produces exactly one of these,
/// `--prompts` one per data row.
///
/// The classification columns are carried verbatim rather than parsed. This
/// binary never branches on them; they exist so the sweep's JSONL can be
/// sliced by prompt class afterwards.
struct Job {
    id: String,
    text: String,
    axis: Option<String>,
    value: Option<String>,
    lang: Option<String>,
    register: Option<String>,
    out: PathBuf,
}

/// Header columns the TSV must have. Everything else is optional, and a
/// column the file does not declare is simply absent from the JSONL.
const REQUIRED_COLUMNS: [&str; 2] = ["id", "prompt"];

/// Classification columns carried through to the JSONL when present.
const CARRIED_COLUMNS: [&str; 4] = ["axis", "value", "lang", "register"];

/// Read the held-out prompt table.
///
/// Columns are addressed by header NAME, never by position, so a file that
/// grows a column later still reads correctly. A trailing newline and blank
/// lines are tolerated; a missing `id` or `prompt` column, a short row, or an
/// empty id or prompt is an error naming the file and the row.
fn parse_prompts_tsv(path: &Path, out_dir: &Path) -> Result<Vec<Job>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("--prompts {}: {e}", path.display()))?;
    let mut lines = text
        .lines()
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty());

    let (_, header_line) = lines
        .next()
        .ok_or_else(|| format!("--prompts {}: file is empty", path.display()))?;
    let header: Vec<&str> = header_line.split('\t').map(str::trim).collect();
    let index_of = |name: &str| header.iter().position(|column| *column == name);

    let missing: Vec<&str> = REQUIRED_COLUMNS
        .iter()
        .copied()
        .filter(|name| index_of(name).is_none())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "--prompts {}: header is missing required column(s) {:?}; it has {:?}",
            path.display(),
            missing,
            header
        ));
    }
    // Both indices exist: `missing` is empty, and it was built from the same
    // lookup. The `ok_or_else` repeats that rather than unwrapping on it.
    let id_at = index_of("id").ok_or("id column vanished between checks")?;
    let prompt_at = index_of("prompt").ok_or("prompt column vanished between checks")?;
    let carried: Vec<(&str, Option<usize>)> = CARRIED_COLUMNS
        .iter()
        .map(|name| (*name, index_of(name)))
        .collect();

    let mut jobs = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (line_number, line) in lines {
        // `enumerate` is zero-based; humans count the header as line 1.
        let row = line_number + 1;
        let fields: Vec<&str> = line.split('\t').collect();
        let field = |at: usize| fields.get(at).map(|value| value.trim());
        let id = field(id_at)
            .ok_or_else(|| {
                format!(
                    "--prompts {} line {row}: {} field(s), need at least {} for the id column",
                    path.display(),
                    fields.len(),
                    id_at + 1
                )
            })?
            .to_string();
        let prompt = field(prompt_at)
            .ok_or_else(|| {
                format!(
                    "--prompts {} line {row}: {} field(s), need at least {} for the prompt column",
                    path.display(),
                    fields.len(),
                    prompt_at + 1
                )
            })?
            .to_string();
        if id.is_empty() {
            return Err(format!("--prompts {} line {row}: empty id", path.display()));
        }
        if prompt.is_empty() {
            return Err(format!(
                "--prompts {} line {row}: empty prompt for id {id}",
                path.display()
            ));
        }
        // Duplicate ids would have the second render overwrite the first wav
        // and leave two contradictory JSONL rows under one key.
        if let Some(first) = seen.insert(id.clone(), row) {
            return Err(format!(
                "--prompts {} line {row}: id {id} already used on line {first}",
                path.display()
            ));
        }

        let mut carried_values: HashMap<&str, Option<String>> = HashMap::new();
        for (name, at) in &carried {
            let value = at
                .and_then(field)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            carried_values.insert(name, value);
        }
        let take = |name: &str| carried_values.get(name).cloned().flatten();

        jobs.push(Job {
            out: out_dir.join(format!("{id}.wav")),
            axis: take("axis"),
            value: take("value"),
            lang: take("lang"),
            register: take("register"),
            id,
            text: prompt,
        });
    }

    if jobs.is_empty() {
        return Err(format!(
            "--prompts {}: header only, no prompt rows",
            path.display()
        ));
    }
    Ok(jobs)
}

/// Build the render list from the parsed command line.
///
/// `--text` yields one job whose id is the output file's stem, so a single
/// render written with `--jsonl` keys the same way a sweep row does.
fn build_jobs(args: &Args) -> Result<Vec<Job>, String> {
    match (&args.text, &args.prompts) {
        (Some(text), None) => {
            let out = args.out.clone().ok_or("--out is required with --text")?;
            let id = out
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("render")
                .to_string();
            Ok(vec![Job {
                id,
                text: text.clone(),
                axis: None,
                value: None,
                lang: None,
                register: None,
                out,
            }])
        }
        (None, Some(prompts)) => {
            let out_dir = args
                .out_dir
                .as_deref()
                .ok_or("--out-dir is required with --prompts")?;
            std::fs::create_dir_all(out_dir)
                .map_err(|e| format!("--out-dir {}: {e}", out_dir.display()))?;
            parse_prompts_tsv(prompts, out_dir)
        }
        // `parse_args` already rejected both-or-neither.
        _ => Err("exactly one of --text and --prompts is required".to_string()),
    }
}

/// Append-only JSONL sink, flushed after every line.
///
/// Buffering to the end would lose the whole log when a sweep dies partway,
/// which is exactly the case the log exists for.
struct JsonlSink {
    file: File,
    path: PathBuf,
}

impl JsonlSink {
    fn create(path: &Path) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("--jsonl {}: {e}", path.display()))?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// Write one object and flush it to the OS.
    ///
    /// Serialization is `serde_json`, already a boostr dependency, so the
    /// escaping of a prompt containing a quote or a tab is the library's
    /// problem rather than this file's.
    fn write(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let line = serde_json::to_string(value)
            .map_err(|e| format!("--jsonl {}: serializing record: {e}", self.path.display()))?;
        writeln!(self.file, "{line}")
            .map_err(|e| format!("--jsonl {}: {e}", self.path.display()))?;
        self.file
            .flush()
            .map_err(|e| format!("--jsonl {}: {e}", self.path.display()))
    }
}

/// The weight source's short name, for the run record.
fn source_format(weights: &Weights) -> &'static str {
    match weights {
        Weights::Checkpoint(_) => "checkpoint",
        Weights::Gguf(_) => "gguf",
        Weights::Tcf(_) => "tcf",
    }
}

/// The weight file or directory, for the run record.
fn source_path(weights: &Weights) -> &Path {
    match weights {
        Weights::Checkpoint(path) | Weights::Gguf(path) | Weights::Tcf(path) => path,
    }
}

/// The encoding most of the file's tensors are stored at, e.g. `Q6AS64T64`
/// for a TCF or `Q6_K` for a GGUF.
///
/// A single-file model mixes encodings — norms and embeddings routinely fall
/// back to a raw type — so the mode over tensor COUNT is reported rather than
/// a single name that would misrepresent the file. `None` for a checkpoint
/// directory, whose tensors carry no encoding beyond their dtype, and `None`
/// whenever the header cannot be read: the run record is instrumentation and
/// must never abort the sweep.
///
/// Both readers parse the header and directory only, no payload byte, so this
/// costs a mmap and a few hundred microseconds against a 26 s load.
fn dominant_encoding(weights: &Weights) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    match weights {
        Weights::Checkpoint(_) => return None,
        Weights::Tcf(path) => {
            let loader = boostr::format::TcfLoader::open(path).ok()?;
            for tensor in loader.tensors() {
                *counts
                    .entry(boostr::format::tcf::encoding_name(tensor.encoding()))
                    .or_default() += 1;
            }
        }
        Weights::Gguf(path) => {
            let gguf = boostr::format::Gguf::open_with_mmap(path, true).ok()?;
            let names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
            for name in names {
                if let Ok(info) = gguf.tensor_info(&name) {
                    *counts.entry(format!("{:?}", info.ggml_type)).or_default() += 1;
                }
            }
        }
    }
    counts
        .into_iter()
        .max_by_key(|(name, count)| (*count, name.clone()))
        .map(|(name, _)| name)
}

/// Median of `values`, or `None` when empty.
fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("F0 values are finite"));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some(0.5 * (values[mid - 1] + values[mid]))
    }
}

/// Median F0 over voiced frames, Hz, via the crate's YIN estimator. `None`
/// when nothing is voiced or the clip is too short to frame.
fn median_f0(samples: &[f32], sample_rate: u32) -> Option<f64> {
    let track = estimate_pitch(samples, sample_rate, PitchOptions::default()).ok()?;
    median(track.f0.iter().flatten().copied().collect())
}

/// Report the reference's median F0, or that it has none.
///
/// Only `--best-of > 1` consults it, so it is only ever printed there. Split
/// out because a sweep prints it once at load and a single render prints it in
/// its historical position, and the two must say the same thing.
fn report_reference_f0(ref_f0: Option<f64>) {
    match ref_f0 {
        Some(hz) => eprintln!("reference median F0: {hz:.1} Hz"),
        None => eprintln!("reference has no measurable pitch; --best-of will keep the first take"),
    }
}

/// One generated candidate.
struct Take {
    seed: u64,
    outcome: GenerateOutcome,
    patches: usize,
    samples: Vec<f32>,
    /// Median F0 in Hz, `None` when the take has no measurable pitch.
    f0: Option<f64>,
}

/// Report one structural check, returning whether it passed.
fn check(label: &str, pass: bool, detail: String) -> bool {
    eprintln!("  [{}] {label}: {detail}", if pass { "OK" } else { "FAIL" });
    pass
}

/// Read a reference recording as mono 16 kHz, also reporting the file's own
/// rate and channel count.
///
/// Uses `decode_audio` plus `to_mono_at_rate` rather than the one-shot
/// `decode_audio_file_mono_at`, which returns only the resampled samples and
/// so cannot report the file's native rate. The two paths run the same
/// symphonia decode and the same resampler.
struct ReferenceAudio {
    /// Mono samples resampled to [`REF_RATE`], what the model consumes.
    samples: Vec<f32>,
    /// The file's own rate and channel count, reported so the operator can see
    /// whether a resample happened.
    native_rate: u32,
    native_channels: u16,
    native_frames: usize,
}

fn load_reference(path: &Path) -> Result<ReferenceAudio, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint)?;
    let (rate, channels, frames) = (data.sample_rate, data.channels, data.frames());
    let mono = to_mono_at_rate(&data, REF_RATE)?;
    Ok(ReferenceAudio {
        samples: mono,
        native_rate: rate,
        native_channels: channels,
        native_frames: frames,
    })
}

/// The model-and-generation body: everything that runs on the chosen
/// runtime `R`, from loading the transformer stack through the structural
/// self-checks, and on through writing each wav.
///
/// The load half runs ONCE and every job in `jobs` reuses it; that is what
/// makes a 66-prompt sweep affordable against a 26 s model load. Returns
/// whether every job passed every structural check. A `--text` run still
/// exits the process on a failed check, unchanged; a sweep records the
/// failure and carries on.
fn run<R: Runtime<DType = DType>>(
    args: &Args,
    device: &R::Device,
    client: &(impl VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static),
    ref_wav: &[f32],
    jobs: &[Job],
    sink: Option<&mut JsonlSink>,
    started: Instant,
) -> Result<bool, Box<dyn std::error::Error>>
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
    // --- model --------------------------------------------------------------
    let mut model = match &args.weights {
        Weights::Checkpoint(dir) => {
            eprintln!("loading {} ...", dir.display());
            VoxCpm2Model::<R>::from_checkpoint(dir, &args.audiovae, device, args.dtype)?
        }
        Weights::Gguf(path) => {
            eprintln!("loading {} ...", path.display());
            VoxCpm2Model::<R>::from_gguf(
                path,
                args.config.as_deref(),
                &args.audiovae,
                device,
                args.dtype,
            )?
        }
        Weights::Tcf(path) => {
            eprintln!("loading {} ...", path.display());
            // `parse_args` already rejected a missing `--config`; this arm
            // repeats the check rather than unwrapping on that invariant.
            let config = args
                .config
                .as_deref()
                .ok_or("--config is required with --tcf")?;
            VoxCpm2Model::<R>::from_tcf(path, config, &args.audiovae, device, args.dtype)?
        }
    };

    // --- LoRA adapter -------------------------------------------------------
    // `apply_lora` must run before `load_lora_named`: it allocates the
    // `lora_a`/`lora_b` Vars that name lookup then resolves against.
    // `check_lora_metadata` runs first, against the file's own
    // `__metadata__`, so a rank/alpha/targets mismatch aborts before the
    // model is mutated at all.
    if let Some(lora_path) = &args.lora {
        let lora_target_names: Vec<String> = args
            .lora_targets
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        eprintln!("loading LoRA adapter {} ...", lora_path.display());
        let mut adapter = SafeTensors::open(lora_path)?;
        check_lora_metadata(
            adapter.metadata(),
            args.lora_rank,
            args.lora_alpha,
            &lora_target_names,
        )?;
        let lora_targets = LoraTargets::new(lora_target_names.clone());
        let adapted = model.apply_lora(&lora_targets, args.lora_rank, args.lora_alpha, device)?;
        let lora_tensors = adapter.load_all::<R>(device)?;
        let loaded = model.load_lora_named(&lora_tensors)?;
        eprintln!(
            "LoRA: targets={lora_target_names:?} rank={} alpha={} -> {adapted} projection(s) \
             adapted, {loaded} tensor(s) loaded",
            args.lora_rank, args.lora_alpha
        );
    }

    let ref_feat = model.encode_reference(client, ref_wav)?;
    let t_ref = ref_feat.shape()[0];
    eprintln!("T_ref: {t_ref} reference patches");

    // Loaded once for the whole run. Re-reading tokenizer.json per prompt is
    // the one avoidable cost a sweep could still be left carrying.
    let tokenizer = load_tokenizer(tokenizer_path(&args.weights, args.config.as_deref())?)?;

    // --- reference pitch, only when it decides something ---------------------
    // Computed once, because the reference does not change between prompts. A
    // sweep reports it here; a single render reports it in its original
    // position below, so `--text` stderr is byte-for-byte what it always was.
    let sweep = args.prompts.is_some();
    let ref_f0 = if args.best_of > 1 {
        median_f0(ref_wav, REF_RATE)
    } else {
        None
    };
    if sweep && args.best_of > 1 {
        report_reference_f0(ref_f0);
    }

    // Everything above is load: weights, adapter, the VAE encode of the
    // reference, the tokenizer. Timed from process start, so it is the number
    // the operator actually waited through, and it is excluded from every
    // render's wall clock.
    let load_seconds = started.elapsed().as_secs_f64();
    let mut sink = sink;
    if let Some(sink) = sink.as_deref_mut() {
        sink.write(&serde_json::json!({
            "record": "run",
            "source_format": source_format(&args.weights),
            "model_path": source_path(&args.weights).display().to_string(),
            "encoding": dominant_encoding(&args.weights),
            "config": args.config.as_ref().map(|p| p.display().to_string()),
            "lora": args.lora.as_ref().map(|p| p.display().to_string()),
            "audiovae": args.audiovae.display().to_string(),
            "reference": args.reference.display().to_string(),
            "prompts": args.prompts.as_ref().map(|p| p.display().to_string()),
            "device": match args.device {
                Device::Cpu => "cpu",
                Device::Cuda => "cuda",
            },
            "dtype": match args.dtype {
                Some(dtype) => format!("{dtype:?}"),
                None => "native".to_string(),
            },
            "n_timesteps": args.n_timesteps,
            "cfg": args.cfg,
            "min_len": args.min_len,
            "best_of": args.best_of,
            "base_seed": args.seed,
            "sample_rate": SAMPLE_RATE,
            "renders": jobs.len(),
            "load_seconds": load_seconds,
        }))?;
    }

    let patch_size = model.config.patch_size;
    let generator = model.patch_generator();
    let mut all_ok = true;

    for (job_index, job) in jobs.iter().enumerate() {
        if sweep {
            eprintln!(
                "--- [{}/{}] {} -> {} ---",
                job_index + 1,
                jobs.len(),
                job.id,
                job.out.display()
            );
        }

        // --- text -----------------------------------------------------------
        let normalized = normalize_whitespace(&job.text);
        let mut text_token_ids = tokenize(&tokenizer, &normalized);
        let text_len = text_token_ids.len();
        // `prefill` rejects a sequence that does not end here: AUDIO_START_ID is
        // the position the first generated patch attends from.
        text_token_ids.push(AUDIO_START_ID);
        let max_len = args.max_len.unwrap_or_else(|| default_max_len(text_len));
        eprintln!("text: {normalized:?}");
        eprintln!(
            "text tokens: {text_len} (+1 AUDIO_START_ID = {}), max_len {max_len}{}",
            text_token_ids.len(),
            if args.max_len.is_some() {
                " (given)"
            } else {
                " (computed: min(text_len * 6 + 10, 4096))"
            }
        );

        // Both KV caches are sized once, for the prefill prefix plus every patch
        // the loop may emit.
        let seq_len = t_ref + 2 + text_token_ids.len();
        let max_length = seq_len + max_len;

        let mut options = GenerateOptions::new(max_len, args.seed);
        options.cfm.n_timesteps = args.n_timesteps;
        options.cfm.cfg_value = args.cfg;
        options.min_len = args.min_len;
        eprintln!(
            "options: n_timesteps={} cfg={} min_len={} max_len={} seed={} best_of={}",
            options.cfm.n_timesteps,
            options.cfm.cfg_value,
            options.min_len,
            options.max_len,
            options.seed,
            args.best_of
        );
        if !sweep && args.best_of > 1 {
            report_reference_f0(ref_f0);
        }

        // --- generate ---------------------------------------------------------
        // `render_started` covers EVERY take, not just the winner. With
        // `--best-of 1` that is one render; above it the wall clock is the cost
        // of the whole selection, and the record says so.
        let render_started = Instant::now();
        // Phase accumulators, summed ACROSS takes so `--best-of 4` reports the
        // cost of four prefills, four loops and four decodes. Prefill is timed
        // separately because it is neither generation nor vocoding: it is the
        // reference features and the text prefix through the stack once, and
        // on CPU it is large enough to be mistaken for the decode if the two
        // are lumped together.
        let mut prefill_seconds = 0.0f64;
        let mut generate_seconds = 0.0f64;
        let mut vocode_seconds = 0.0f64;
        let mut pitch_seconds = 0.0f64;
        let mut takes = Vec::with_capacity(args.best_of);

        for take_index in 0..args.best_of {
            let seed = args.seed + take_index as u64;
            options.seed = seed;
            eprintln!("take {}/{} (seed {seed}) ...", take_index + 1, args.best_of);

            // The prefill is re-run per take: GenerateState::start consumes the
            // PrefillState and the loop advances its KV caches in place.
            let prefill_started = Instant::now();
            let prefill = model.prefill(client, Some(&ref_feat), &text_token_ids, max_length)?;
            let mut state = GenerateState::start(prefill, model.config)?;
            prefill_seconds += prefill_started.elapsed().as_secs_f64();

            // Mirrors PatchGenerator::generate exactly (cap first, then step,
            // Stopped means the stop token fired past min_len). Written out here
            // only so progress can be reported; generate() runs to completion
            // with no callback.
            let take_started = Instant::now();
            let outcome = loop {
                if state.patches.len() >= options.max_len {
                    break GenerateOutcome::MaxLen;
                }
                let step = generator.step(client, &mut state, &options)?;
                let emitted = state.patches.len();
                if emitted % PROGRESS_EVERY == 0 {
                    eprintln!(
                        "  {emitted}/{} patches ({:.1}s)",
                        options.max_len,
                        take_started.elapsed().as_secs_f64()
                    );
                }
                if step == StepOutcome::Stopped {
                    break GenerateOutcome::StopToken;
                }
            };

            generate_seconds += take_started.elapsed().as_secs_f64();

            // The AudioVAE decode answers to a different optimization story
            // than the loop above. A CUDA profile of this model put
            // `tcf_gemm_f32` (the loop) at 74.7% of GPU time and
            // `conv1d_oc4_f32` (this) at 10.7%, mean 15.2 ms and max 117 ms
            // per call. Those SHARES predate the register-blocked GEMM, which
            // cut the loop four to six times over; the VAE's absolute cost is
            // what the shares were measuring and it has not moved. The VAE
            // is loaded from `.pth` at full precision whatever tier the
            // language model came from, so its cost is a CONSTANT across the
            // matrix and dilutes any tier comparison it is folded into.
            let vocode_started = Instant::now();
            let patches = state.patches.len();
            let decoded = model.decode_patches(client, &state.patches)?;
            let samples: Vec<f32> = decoded.contiguous()?.to_vec();
            vocode_seconds += vocode_started.elapsed().as_secs_f64();
            eprintln!(
                "  {} after {patches} patches, {} samples ({:.2}s audio, {:.1}s wall)",
                match outcome {
                    GenerateOutcome::StopToken => "STOP TOKEN",
                    GenerateOutcome::MaxLen => "MAX LEN (truncated)",
                },
                samples.len(),
                samples.len() as f64 / SAMPLE_RATE as f64,
                take_started.elapsed().as_secs_f64()
            );

            let pitch_started = Instant::now();
            let f0 = if ref_f0.is_some() {
                let f0 = median_f0(&samples, SAMPLE_RATE as u32);
                match f0 {
                    Some(hz) => eprintln!("  median F0 {hz:.1} Hz"),
                    None => eprintln!("  no measurable pitch"),
                }
                f0
            } else {
                None
            };
            pitch_seconds += pitch_started.elapsed().as_secs_f64();

            takes.push(Take {
                seed,
                outcome,
                patches,
                samples,
                f0,
            });
        }

        // --- pick the take ----------------------------------------------------
        // A take with no measurable pitch scores infinity, so it never wins.
        let chosen = match ref_f0 {
            Some(reference_hz) => {
                let err = |t: &Take| t.f0.map_or(f64::INFINITY, |hz| (hz - reference_hz).abs());
                takes
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        err(a).partial_cmp(&err(b)).expect("errors are comparable")
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            None => 0,
        };
        let take = &takes[chosen];
        if args.best_of > 1 {
            let detail = match (ref_f0, take.f0) {
                (Some(reference_hz), Some(hz)) => {
                    format!(
                        ", median F0 {hz:.1} Hz, off by {:.1} Hz",
                        (hz - reference_hz).abs()
                    )
                }
                _ => String::new(),
            };
            eprintln!(
                "chosen: take {}/{} (seed {}){detail}",
                chosen + 1,
                args.best_of,
                take.seed
            );
        }

        // --- speed ------------------------------------------------------------
        // `audio_seconds` is the winner's alone; `wall_seconds` covers the
        // whole selection, so an `rtf` from `--best-of 4` is four renders' cost
        // against one render's audio and is only comparable to another
        // best-of-4.
        //
        // `rtf` is the honest end-to-end cost. `generate_rtf` is the one that
        // responds to the quantization tier: the tier changes the weights the
        // step loop multiplies against and nothing else, while prefill scales
        // with the reference and the prompt, and the vocoder is the same
        // full-precision `.pth` in every row of the matrix.
        //
        // `wall_seconds` ends HERE, before the checks and the wav write. The
        // write is timed on its own below and is deliberately NOT inside
        // `wall_seconds`, because a failed single render exits without writing
        // at all and a wall clock that changed meaning with the outcome would
        // be worse than one that excludes a constant.
        let wall_seconds = render_started.elapsed().as_secs_f64();
        let audio_seconds = take.samples.len() as f64 / SAMPLE_RATE as f64;
        let rtf = wall_seconds / audio_seconds;
        let generate_rtf = generate_seconds / audio_seconds;
        eprintln!(
            "rtf {rtf:.3} ({wall_seconds:.1}s wall / {audio_seconds:.2}s audio{})",
            if args.best_of > 1 {
                ", wall covers every take"
            } else {
                ""
            }
        );
        eprintln!(
            "  phases: prefill {prefill_seconds:.2}s, generate {generate_seconds:.2}s \
             (rtf {generate_rtf:.3}), vocode {vocode_seconds:.2}s, pitch {pitch_seconds:.2}s"
        );

        // --- structural self-checks -------------------------------------------
        // f32::max ignores a NaN operand, so NaN cannot hide behind the peak; the
        // separate finite check is what catches it. An infinity does propagate, so
        // the `peak <= 1.0` check catches that on its own.
        let peak = take.samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
        let non_finite = take.samples.iter().filter(|s| !s.is_finite()).count();
        let expected_samples = take.patches * patch_size * HOP_LENGTH;

        eprintln!("checks:");
        let mut ok = true;
        ok &= check(
            "outcome",
            matches!(
                take.outcome,
                GenerateOutcome::StopToken | GenerateOutcome::MaxLen
            ),
            format!("{:?}", take.outcome),
        );
        if take.outcome == GenerateOutcome::StopToken {
            ok &= check(
                "stop past min_len",
                take.patches > options.min_len + 1,
                format!(
                    "{} patches, need > min_len + 1 = {}",
                    take.patches,
                    options.min_len + 1
                ),
            );
        }
        ok &= check(
            "sample count",
            take.samples.len() == expected_samples,
            format!(
                "{} == {} patches * {patch_size} patch_size * {HOP_LENGTH} decoder hop \
                 = {expected_samples}",
                take.samples.len(),
                take.patches
            ),
        );
        ok &= check(
            "finite",
            non_finite == 0,
            format!("{non_finite} non-finite samples of {}", take.samples.len()),
        );
        ok &= check("peak <= 1.0", peak <= 1.0, format!("peak {peak:.6}"));
        ok &= check(
            "not silence",
            peak > SILENCE_EPSILON,
            format!("peak {peak:.6} > {SILENCE_EPSILON:e}"),
        );

        // A failed single render exits before writing, as it always has. A
        // failed sweep render still gets its wav: the fastest way to diagnose a
        // silent or clipped take is to listen to it.
        let wav_written = ok || sweep;
        let mut write_seconds = 0.0f64;
        if wav_written {
            let write_started = Instant::now();
            let wav = encode_wav_pcm16(&take.samples, SAMPLE_RATE as u32)?;
            std::fs::write(&job.out, wav)?;
            write_seconds = write_started.elapsed().as_secs_f64();
            eprintln!(
                "wrote {} ({audio_seconds:.2}s at {SAMPLE_RATE} Hz, peak {peak:.4})",
                job.out.display()
            );
        }

        if let Some(sink) = sink.as_deref_mut() {
            sink.write(&serde_json::json!({
                "record": "render",
                "id": job.id,
                "text": job.text,
                "axis": job.axis,
                "value": job.value,
                "lang": job.lang,
                "register": job.register,
                "seed": take.seed,
                "base_seed": args.seed,
                "best_of": args.best_of,
                "take": chosen + 1,
                "out_path": job.out.display().to_string(),
                "wav_written": wav_written,
                "wall_seconds": wall_seconds,
                "prefill_seconds": prefill_seconds,
                "generate_seconds": generate_seconds,
                "vocode_seconds": vocode_seconds,
                "pitch_seconds": pitch_seconds,
                "write_seconds": write_seconds,
                "audio_seconds": audio_seconds,
                "rtf": rtf,
                "generate_rtf": generate_rtf,
                "patches": take.patches,
                "samples": take.samples.len(),
                "text_tokens": text_len,
                "max_len": max_len,
                "stop_reason": match take.outcome {
                    GenerateOutcome::StopToken => "stop_token",
                    GenerateOutcome::MaxLen => "max_len",
                },
                "checks_passed": ok,
                "peak": peak,
                "non_finite": non_finite,
                "f0_hz": take.f0,
                "ref_f0_hz": ref_f0,
            }))?;
        }

        if !ok {
            all_ok = false;
            if !sweep {
                eprintln!("FAILED after {:.1}s", started.elapsed().as_secs_f64());
                std::process::exit(1);
            }
            eprintln!(
                "FAILED {} after {:.1}s (continuing)",
                job.id,
                started.elapsed().as_secs_f64()
            );
        }
    }

    Ok(all_ok)
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

    // --- render list --------------------------------------------------------
    // Built before anything is loaded: a malformed TSV must not cost a 26 s
    // model load before it is reported.
    let jobs = match build_jobs(&args) {
        Ok(jobs) => jobs,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let mut sink = match args.jsonl.as_deref().map(JsonlSink::create).transpose() {
        Ok(sink) => sink,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    if args.prompts.is_some() {
        eprintln!("sweep: {} prompt(s)", jobs.len());
    }

    // --- reference audio ----------------------------------------------------
    let reference = load_reference(&args.reference)?;
    let ReferenceAudio {
        samples: ref_wav,
        native_rate,
        native_channels: channels,
        native_frames,
    } = reference;
    eprintln!(
        "reference: {} ({:.2}s, {native_rate} Hz, {channels} ch) -> {} samples at {REF_RATE} Hz",
        args.reference.display(),
        native_frames as f64 / native_rate as f64,
        ref_wav.len()
    );

    let all_ok = match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime>(
                &args,
                &device,
                &client,
                &ref_wav,
                &jobs,
                sink.as_mut(),
                started,
            )?
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run::<CudaRuntime>(
                &args,
                &device,
                &client,
                &ref_wav,
                &jobs,
                sink.as_mut(),
                started,
            )?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            eprintln!(
                "--device cuda: this binary was built without CUDA support; rebuild with \
                 --features cuda"
            );
            std::process::exit(2);
        }
    };

    eprintln!("total {:.1}s", started.elapsed().as_secs_f64());
    // A single render already exited on a failed check. A sweep gets here with
    // its remaining renders done and its log complete, and fails at the end so
    // the exit status still reports the bad row.
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
