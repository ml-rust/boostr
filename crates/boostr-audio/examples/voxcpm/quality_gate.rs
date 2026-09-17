//! Score a batch of VoxCPM2 renders: intelligibility, pace, signal, timbre.
//!
//! ```text
//! cargo run --release --features voxcpm,whisper,vad,cuda,f16 \
//!     --example voxcpm_quality_gate -- \
//!     (--renders LOG.jsonl | --manifest ROWS.tsv) \
//!     --whisper WHISPER_DIR --vad silero_vad_16k.safetensors \
//!     [--audiovae audiovae.pth [--ref REF.wav]] \
//!     [--device cpu|cuda] [--language ms|en|auto-from-row] \
//!     [--out scores.jsonl] \
//!     [--max-wer X] [--max-cer X] [--min-hnr-db X] [--min-lufs X] [--max-lufs X] \
//!     [--max-leading-silence-s X] [--max-trailing-silence-s X] \
//!     [--min-words-per-s X] [--max-words-per-s X] [--min-latent-cosine X]
//! ```
//!
//! # Inputs
//!
//! `--renders` is the `voxcpm_clone --prompts … --jsonl` log. Each
//! `"record":"render"` row supplies `id`, `text`, `lang`, `axis`, `out_path`,
//! `audio_seconds`, `stop_reason`, `checks_passed` and `ref_f0_hz`; the
//! `"record":"run"` row supplies `reference`, `model_path`, `lora`,
//! `n_timesteps` and `cfg`, which the summary carries as `source`. A relative
//! `out_path` is tried as given, then relative to the log's directory.
//!
//! `--manifest` is a header-named TSV with `id`, `wav`, `text` and optional
//! `lang`, `axis`, `ref_wav` columns, for renders that came from elsewhere.
//! Paths resolve against the manifest's directory.
//!
//! # Metrics, and what they cannot tell you
//!
//! - `wer`, `cer`: Whisper transcript against the prompt. Catches dropped,
//!   invented and slurred words. It SATURATES on clean TTS: Whisper reads a
//!   mediocre render word-perfect, so a batch at zero WER is a floor reached,
//!   not a ranking (`audio/eval/n_timesteps_ab/README` shows exactly that).
//! - `hnr_db`: harmonic-to-noise ratio under the voice. Buzz, hiss and
//!   vocoder artefacts drag it down. `snr_db` and `floor_dbfs` only see the
//!   pauses.
//! - `lufs`, `peak_dbfs`, `clipped_samples`: level and headroom.
//! - `leading_silence_s`, `trailing_silence_s`, `speech_ratio`: the silence
//!   budget around the speech.
//! - `words_per_s`: PROMPT words per second of VAD speech. Words come from
//!   what was asked, not from the transcript, so a truncated render reads
//!   high and a stalled or babbling one reads low. Pair with `wer` to tell
//!   which.
//! - `latent_cosine`, `f0_mean_delta_hz`, `f0_std_ratio`: drift against the
//!   reference clip. The cosine is over time-pooled AudioVAE latents, a
//!   reconstruction proxy, NOT a speaker-verification score. It ranks two
//!   models on the same speaker and prompts; it does not say "same person".
//!
//! Naturalness has no single number here. HNR, pace and timbre together
//! carry it; WER does not.
//!
//! # Output
//!
//! One `"record":"score"` object per row on `--out` (every metric flattened
//! under stable snake_case names, plus `transcript` and `violations`), then
//! one `"record":"summary"` with per-axis and overall corpus WER/CER, mean
//! and median of every metric, and a count of rows failing each threshold. A
//! readable table goes to stderr. Exit status is 1 when any row has a
//! violation or could not be scored, 0 otherwise; with no threshold flags
//! there are no violations, so the gate only fails on unreadable rows.
//!
//! # Language
//!
//! Whisper's language token changes what it emits. `auto-from-row` (the
//! default) takes each row's `lang`, mapping `mix` to `ms` because code-
//! switched Malay transcribes under `ms` and collapses under `en`; a row
//! with no `lang` falls back to `ms`, the corpus this repository is built
//! around. Any other value is passed through for every row.
//!
//! # The recipe
//!
//! 1. Render: `voxcpm_clone --gguf MODEL.gguf --ref REF.wav --prompts
//!    audio/eval/heldout_prompts.tsv --out-dir renders/ --jsonl renders.jsonl
//!    --device cuda`, once per model or adapter under test.
//! 2. Gate: this binary with `--renders renders.jsonl --out scores.jsonl`,
//!    the same Whisper, VAD, AudioVAE and `--ref` for every set.
//! 3. Compare: the `"record":"summary"` lines of two `scores.jsonl` files by
//!    axis. Per-row rows join on `id`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use boostr::model::audio::vad::SileroVad;
use boostr::model::audio::voxcpm::VoxCpmClient;
use boostr_audio::VadSegmentOptions;
use boostr_audio::tts_eval::{
    F0Stats, IntelligibilityClient, IntelligibilityScorer, RowScore, Summary, Thresholds,
    TimbreScorer, pace, signal_stats, timbre_proxy, violations,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConvOps, ReduceOps, ScalarOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

// Sibling modules: the shared JSONL sink, the two input readers, and the
// stderr report.
mod gate_inputs;
mod gate_report;
mod jsonl_sink;
use gate_inputs::{
    AUTO_LANGUAGE, Clip, InputRow, language_for, load_clip, load_manifest, load_renders,
};
use gate_report::{report_row, report_summary, source_json};
use jsonl_sink::JsonlSink;

const USAGE: &str = "usage: voxcpm_quality_gate (--renders LOG.jsonl | --manifest ROWS.tsv) \
--whisper DIR --vad FILE.safetensors [--audiovae PATH [--ref REF.wav]] \
[--device cpu|cuda] [--language ms|en|auto-from-row] [--out scores.jsonl] \
[--max-wer X] [--max-cer X] [--min-hnr-db X] [--min-lufs X] [--max-lufs X] \
[--max-leading-silence-s X] [--max-trailing-silence-s X] \
[--min-words-per-s X] [--max-words-per-s X] [--min-latent-cosine X]";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Cuda,
}

fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => Err(format!("unknown --device {other}, expected cpu or cuda")),
    }
}

enum Input {
    Renders(PathBuf),
    Manifest(PathBuf),
}

pub struct Args {
    input: Input,
    pub whisper: PathBuf,
    pub vad: PathBuf,
    pub audiovae: Option<PathBuf>,
    pub reference: Option<PathBuf>,
    device: Device,
    pub language: String,
    out: Option<PathBuf>,
    thresholds: Thresholds,
}

fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn take_f64(argv: &[String], i: &mut usize, flag: &str) -> Result<f64, String> {
    take_value(argv, i, flag)?
        .parse::<f64>()
        .map_err(|e| format!("{flag}: {e}"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut renders = None;
    let mut manifest = None;
    let mut whisper = None;
    let mut vad = None;
    let mut audiovae = None;
    let mut reference = None;
    let mut device = Device::Cpu;
    let mut language = AUTO_LANGUAGE.to_string();
    let mut out = None;
    let mut t = Thresholds::default();
    let mut min_wps = None;
    let mut max_wps = None;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--renders" => renders = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--whisper" => whisper = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--vad" => vad = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audiovae" => audiovae = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--ref" => reference = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--device" => device = parse_device(&take_value(&argv, &mut i, flag)?)?,
            "--language" => language = take_value(&argv, &mut i, flag)?,
            "--out" => out = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--max-wer" => t.max_wer = Some(take_f64(&argv, &mut i, flag)?),
            "--max-cer" => t.max_cer = Some(take_f64(&argv, &mut i, flag)?),
            "--min-hnr-db" => t.min_hnr_db = Some(take_f64(&argv, &mut i, flag)?),
            "--min-lufs" => t.min_lufs = Some(take_f64(&argv, &mut i, flag)?),
            "--max-lufs" => t.max_lufs = Some(take_f64(&argv, &mut i, flag)?),
            "--max-leading-silence-s" => {
                t.max_leading_silence_s = Some(take_f64(&argv, &mut i, flag)?)
            }
            "--max-trailing-silence-s" => {
                t.max_trailing_silence_s = Some(take_f64(&argv, &mut i, flag)?)
            }
            "--min-words-per-s" => min_wps = Some(take_f64(&argv, &mut i, flag)?),
            "--max-words-per-s" => max_wps = Some(take_f64(&argv, &mut i, flag)?),
            "--min-latent-cosine" => t.min_latent_cosine = Some(take_f64(&argv, &mut i, flag)?),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }
    if min_wps.is_some() || max_wps.is_some() {
        t.words_per_s = Some((min_wps.unwrap_or(0.0), max_wps.unwrap_or(f64::INFINITY)));
    }
    let input = match (renders, manifest) {
        (Some(r), None) => Input::Renders(r),
        (None, Some(m)) => Input::Manifest(m),
        (Some(_), Some(_)) => {
            return Err(format!(
                "--renders and --manifest are mutually exclusive\n{USAGE}"
            ));
        }
        (None, None) => return Err(format!("--renders or --manifest is required\n{USAGE}")),
    };
    if reference.is_some() && audiovae.is_none() {
        return Err(format!("--ref needs --audiovae\n{USAGE}"));
    }
    Ok(Args {
        input,
        whisper: whisper.ok_or_else(|| format!("--whisper is required\n{USAGE}"))?,
        vad: vad.ok_or_else(|| format!("--vad is required\n{USAGE}"))?,
        audiovae,
        reference,
        device,
        language,
        out,
        thresholds: t,
    })
}

/// Reference embeddings by path, so one `--ref` is encoded once.
type RefCache = HashMap<PathBuf, (Vec<f32>, F0Stats)>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let (run, rows) = match &args.input {
        Input::Renders(p) => load_renders(p),
        Input::Manifest(p) => load_manifest(p).map(|rows| (serde_json::Value::Null, rows)),
    }
    .unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    let mut sink = match args.out.as_deref().map(JsonlSink::create).transpose() {
        Ok(sink) => sink,
        Err(message) => {
            eprintln!("--out {message}");
            std::process::exit(2);
        }
    };

    let ok = match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run_gate::<CpuRuntime, _>(&args, &device, &client, &run, &rows, sink.as_mut())?
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run_gate::<CudaRuntime, _>(&args, &device, &client, &run, &rows, sink.as_mut())?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            eprintln!(
                "--device cuda: this binary was built without CUDA; rebuild with --features cuda"
            );
            std::process::exit(2);
        }
    };
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

/// Load every model once, score every row, write the summary. Returns
/// whether the gate passed.
fn run_gate<R, C>(
    args: &Args,
    device: &R::Device,
    client: &C,
    run: &serde_json::Value,
    rows: &[InputRow],
    mut sink: Option<&mut JsonlSink>,
) -> Result<bool, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: IntelligibilityClient<R> + VoxCpmClient<R>,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + ReduceOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>,
{
    eprintln!("loading {} ...", args.whisper.display());
    let long_clip_vad = SileroVad::<R>::from_safetensors(&args.vad, device)?;
    let whisper = IntelligibilityScorer::<R>::from_dir(&args.whisper, device, client)?
        .with_vad(long_clip_vad);
    let vad = SileroVad::<R>::from_safetensors(&args.vad, device)?;
    let timbre = args
        .audiovae
        .as_deref()
        .map(|p| TimbreScorer::<R>::from_checkpoint(p, device))
        .transpose()?;
    let vad_opts = VadSegmentOptions::default();
    let mut refs: RefCache = HashMap::new();

    let mut scored: Vec<RowScore> = Vec::with_capacity(rows.len());
    for input in rows {
        let mut row = RowScore::new(&input.id, &input.text, input.wav.display().to_string());
        row.axis = input.axis.clone();
        row.lang = input.lang.clone();
        row.audio_seconds = input.audio_seconds;
        row.stop_reason = input.stop_reason.clone();
        row.checks_passed = input.checks_passed;
        row.ref_f0_hz = input.ref_f0_hz;
        let mut errors: Vec<String> = Vec::new();

        match load_clip(&input.wav) {
            Err(e) => errors.push(e),
            Ok(clip) => {
                match signal_stats(&clip.native, clip.native_rate) {
                    Ok(s) => row.signal = Some(s),
                    Err(e) => errors.push(format!("signal: {e}")),
                }
                match pace(&vad, client, &clip.at_16k, &input.text, &vad_opts) {
                    Ok(p) => row.pace = Some(p),
                    Err(e) => errors.push(format!("pace: {e}")),
                }
                let language = language_for(&args.language, input);
                match whisper.score(client, &clip.at_16k, &input.text, Some(&language)) {
                    Ok(i) => row.intelligibility = Some(i),
                    Err(e) => errors.push(format!("whisper: {e}")),
                }
                let ref_path = input.ref_wav.clone().or_else(|| args.reference.clone());
                if let (Some(scorer), Some(ref_path)) = (&timbre, ref_path) {
                    match score_timbre(scorer, client, &mut refs, &ref_path, &clip, row.signal) {
                        Ok(t) => row.timbre = Some(t),
                        Err(e) => errors.push(format!("timbre: {e}")),
                    }
                }
            }
        }
        if !errors.is_empty() {
            row.error = Some(errors.join("; "));
        }
        row.violations = violations(&row, &args.thresholds);
        report_row(&row);
        if let Some(sink) = sink.as_deref_mut() {
            sink.write(&row.to_json())?;
        }
        scored.push(row);
    }

    let summary = Summary::from_rows(&scored);
    let mut summary_json = summary.to_json();
    if let Some(obj) = summary_json.as_object_mut() {
        obj.insert("source".into(), source_json(run, args));
    }
    if let Some(sink) = sink {
        sink.write(&summary_json)?;
    }
    report_summary(&summary, &args.thresholds);
    Ok(summary.rows_with_violations == 0 && summary.errors == 0)
}

/// Embed the render and the (cached) reference, then combine.
fn score_timbre<R, C>(
    scorer: &TimbreScorer<R>,
    client: &C,
    refs: &mut RefCache,
    ref_path: &Path,
    clip: &Clip,
    render_signal: Option<boostr_audio::SignalStats>,
) -> Result<boostr_audio::TimbreProxy, String>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R>,
{
    if !refs.contains_key(ref_path) {
        let reference = load_clip(ref_path)?;
        let embed = scorer
            .embed(client, &reference.at_16k)
            .map_err(|e| format!("{}: {e}", ref_path.display()))?;
        let stats = signal_stats(&reference.native, reference.native_rate)
            .map_err(|e| format!("{}: {e}", ref_path.display()))?;
        refs.insert(
            ref_path.to_path_buf(),
            (
                embed,
                F0Stats {
                    mean_hz: stats.f0_mean_hz,
                    std_hz: stats.f0_std_hz,
                },
            ),
        );
    }
    let (ref_embed, ref_f0) = refs
        .get(ref_path)
        .ok_or_else(|| format!("{}: reference not cached", ref_path.display()))?;
    let render_embed = scorer
        .embed(client, &clip.at_16k)
        .map_err(|e| e.to_string())?;
    let render_f0 = F0Stats {
        mean_hz: render_signal.and_then(|s| s.f0_mean_hz),
        std_hz: render_signal.and_then(|s| s.f0_std_hz),
    };
    timbre_proxy(&render_embed, ref_embed, render_f0, *ref_f0).map_err(|e| e.to_string())
}
