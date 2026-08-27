//! End-to-end VoxCPM2 voice cloning: reference wav plus target text in, a
//! 48 kHz wav out. This is the Rust replacement for
//! `audio/pipeline/voxcpm_clone.py`.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_clone -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf [--config config.json]) \
//!     --audiovae audiovae.safetensors \
//!     --ref REF.wav --text "..." --out OUT.wav \
//!     [--n-timesteps 32] [--cfg 2.0] [--min-len 2] [--max-len N] \
//!     [--seed 0] [--best-of N] [--dtype f32] [--device cpu]
//! ```
//!
//! `CKPT_DIR` holds `config.json`, `model.safetensors` and `tokenizer.json`.
//! `--audiovae` is the separately converted `audiovae.safetensors` written by
//! `audio/pipeline/convert_audiovae.py`.
//!
//! `--gguf` is the single-file alternative, written by `compressr convert
//! CKPT_DIR --format gguf --quantization q4_k`. It is mutually exclusive with
//! `--ckpt`. A GGUF carries the transformer stack ONLY, so `--audiovae` is
//! still required, `--config` supplies the `config.json` the file has no
//! embedded copy of, and `tokenizer.json` is looked for beside the `.gguf`
//! and then beside `--config`. The weights arrive DEQUANTIZED to F32, so a
//! Q4_K file is smaller on disk but not in memory.
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

use std::path::{Path, PathBuf};
use std::time::Instant;

use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateOutcome, GenerateState, StepOutcome, VoxCpm2Model,
};
use boostr::model::audio::voxcpm::vae::decoder::{HOP_LENGTH, SAMPLE_RATE};
use boostr::model::audio::voxcpm::{VoxCpmClient, load_tokenizer, normalize_whitespace, tokenize};
use boostr::model::audio::{
    PitchOptions, decode_audio, encode_wav_pcm16, estimate_pitch, extension_hint, to_mono_at_rate,
};
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

/// Where the transformer stack's weights come from.
///
/// `--ckpt` names a checkpoint DIRECTORY (`config.json`,
/// `model.safetensors`, `tokenizer.json`); `--gguf` names a single file that
/// carries the weights and nothing else. Mutually exclusive, and one of them
/// is required.
enum Weights {
    Checkpoint(PathBuf),
    Gguf(PathBuf),
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
    /// `config.json` for the GGUF path. Ignored for `--ckpt`, which reads the
    /// one in the checkpoint directory.
    config: Option<PathBuf>,
    audiovae: PathBuf,
    reference: PathBuf,
    text: String,
    out: PathBuf,
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

const USAGE: &str = "usage: voxcpm_clone (--ckpt DIR | --gguf MODEL.gguf [--config config.json]) \
--audiovae PATH --ref REF.wav \
--text \"...\" --out OUT.wav [--n-timesteps 32] [--cfg 2.0] [--min-len 2] \
[--max-len N] [--seed 0] [--best-of 1] [--dtype f32|bf16|f16|native] \
[--device cpu|cuda]";

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
    let mut config: Option<PathBuf> = None;
    let mut n_timesteps = 32usize;
    let mut cfg = 2.0f32;
    let mut min_len = 2usize;
    let mut max_len = None;
    let mut seed = 0u64;
    let mut best_of = 1usize;
    let mut dtype = Some(DType::F32);
    let mut device = Device::Cpu;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--ckpt" => ckpt = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--gguf" => gguf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--config" => config = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audiovae" => audiovae = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--ref" => reference = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--text" => text = Some(take_value(&argv, &mut i, flag)?),
            "--out" => out = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
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

    // Exactly one weight source. Accepting both and silently preferring one
    // would load a different model than the operator asked for.
    let weights = match (ckpt, gguf) {
        (Some(_), Some(_)) => {
            return Err(format!("--ckpt and --gguf are mutually exclusive\n{USAGE}"));
        }
        (Some(dir), None) => Weights::Checkpoint(dir),
        (None, Some(path)) => Weights::Gguf(path),
        (None, None) => return Err(format!("--ckpt or --gguf is required\n{USAGE}")),
    };

    Ok(Args {
        weights,
        config,
        audiovae: audiovae.ok_or_else(|| format!("--audiovae is required\n{USAGE}"))?,
        reference: reference.ok_or_else(|| format!("--ref is required\n{USAGE}"))?,
        text: text.ok_or_else(|| format!("--text is required\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n{USAGE}"))?,
        n_timesteps,
        cfg,
        min_len,
        max_len,
        seed,
        best_of,
        dtype,
        device,
    })
}

/// Locate `tokenizer.json`.
///
/// A checkpoint directory holds it outright. A GGUF carries no tokenizer at
/// all, so it is looked for beside the `.gguf` first and beside `--config`
/// second — both of those normally sit in, or are copied from, the same
/// checkpoint directory. Neither: an error, rather than a tokenizer guess
/// that would silently produce the wrong token ids.
fn tokenizer_path(weights: &Weights, config: Option<&Path>) -> Result<PathBuf, String> {
    match weights {
        Weights::Checkpoint(dir) => Ok(dir.join("tokenizer.json")),
        Weights::Gguf(path) => {
            let beside = |p: &Path| {
                p.parent()
                    .map(|dir| dir.join("tokenizer.json"))
                    .filter(|candidate| candidate.is_file())
            };
            beside(path)
                .or_else(|| config.and_then(beside))
                .ok_or_else(|| {
                    format!(
                        "no tokenizer.json beside {} (a GGUF carries none); put it there \
                     or pass --config pointing into the checkpoint directory",
                        path.display()
                    )
                })
        }
    }
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
/// self-checks. Returns the chosen take's waveform samples and their peak
/// absolute value, which is all `main`'s write step and closing report
/// need. Exits the process on a failed structural check, same as the
/// single-runtime version this was split out of.
fn run<R: Runtime<DType = DType>>(
    args: &Args,
    device: &R::Device,
    client: &(impl VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R>),
    ref_wav: &[f32],
    started: Instant,
) -> Result<(Vec<f32>, f32), Box<dyn std::error::Error>>
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
        + TypeConversionOps<R>,
{
    // --- model --------------------------------------------------------------
    let model = match &args.weights {
        Weights::Checkpoint(dir) => {
            eprintln!("loading {} ...", dir.display());
            VoxCpm2Model::<R>::from_checkpoint(dir, &args.audiovae, device, args.dtype)?
        }
        Weights::Gguf(path) => {
            eprintln!("loading {} (dequantizing to F32) ...", path.display());
            VoxCpm2Model::<R>::from_gguf(
                path,
                args.config.as_deref(),
                &args.audiovae,
                device,
                args.dtype,
            )?
        }
    };

    let ref_feat = model.encode_reference(client, ref_wav)?;
    let t_ref = ref_feat.shape()[0];
    eprintln!("T_ref: {t_ref} reference patches");

    // --- text ---------------------------------------------------------------
    let tokenizer = load_tokenizer(tokenizer_path(&args.weights, args.config.as_deref())?)?;
    let normalized = normalize_whitespace(&args.text);
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

    // --- reference pitch, only when it decides something ---------------------
    let ref_f0 = if args.best_of > 1 {
        let f0 = median_f0(ref_wav, REF_RATE);
        match f0 {
            Some(hz) => eprintln!("reference median F0: {hz:.1} Hz"),
            None => {
                eprintln!("reference has no measurable pitch; --best-of will keep the first take")
            }
        }
        f0
    } else {
        None
    };

    // --- generate -----------------------------------------------------------
    let patch_size = model.config.patch_size;
    let generator = model.patch_generator();
    let mut takes = Vec::with_capacity(args.best_of);

    for take_index in 0..args.best_of {
        let seed = args.seed + take_index as u64;
        options.seed = seed;
        eprintln!("take {}/{} (seed {seed}) ...", take_index + 1, args.best_of);

        // The prefill is re-run per take: GenerateState::start consumes the
        // PrefillState and the loop advances its KV caches in place.
        let prefill = model.prefill(client, &ref_feat, &text_token_ids, max_length)?;
        let mut state = GenerateState::start(prefill, model.config)?;

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

        let patches = state.patches.len();
        let decoded = model.decode_patches(client, &state.patches)?;
        let samples: Vec<f32> = decoded.contiguous()?.to_vec();
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

        takes.push(Take {
            seed,
            outcome,
            patches,
            samples,
            f0,
        });
    }

    // --- pick the take ------------------------------------------------------
    // A take with no measurable pitch scores infinity, so it never wins.
    let chosen = match ref_f0 {
        Some(reference_hz) => takes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let err = |t: &Take| t.f0.map_or(f64::INFINITY, |hz| (hz - reference_hz).abs());
                err(a).partial_cmp(&err(b)).expect("errors are comparable")
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
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

    // --- structural self-checks ---------------------------------------------
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
            "{} == {} patches * {patch_size} patch_size * {HOP_LENGTH} decoder hop = {expected_samples}",
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

    if !ok {
        eprintln!("FAILED after {:.1}s", started.elapsed().as_secs_f64());
        std::process::exit(1);
    }

    // `chosen` is always a valid index: it is either the literal `0` or a
    // position `min_by` found by iterating `takes` itself.
    Ok((takes.swap_remove(chosen).samples, peak))
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

    let (samples, peak) = match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime>(&args, &device, &client, &ref_wav, started)?
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run::<CudaRuntime>(&args, &device, &client, &ref_wav, started)?
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

    // --- write --------------------------------------------------------------
    let wav = encode_wav_pcm16(&samples, SAMPLE_RATE as u32)?;
    std::fs::write(&args.out, wav)?;
    eprintln!(
        "wrote {} ({:.2}s at {SAMPLE_RATE} Hz, peak {peak:.4})",
        args.out.display(),
        samples.len() as f64 / SAMPLE_RATE as f64
    );
    eprintln!("total {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}
