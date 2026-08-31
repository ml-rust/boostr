//! Transcribe an audio file with Whisper, in pure Rust.
//!
//! ```text
//! cargo run --release --features audio --example transcribe -- \
//!     --model WHISPER_DIR --audio FILE [--audio FILE ...] [--audio-dir DIR] \
//!     [--device cuda] [--language ms] [--translate]
//! ```
//!
//! `--device` selects the runtime and defaults to `cpu`. `cuda` needs a binary
//! built with `--features cuda`; large-v3 on CPU is minutes per batch where the
//! GPU is seconds, so pass it whenever the build has it.
//!
//! `--audio` may be repeated and `--audio-dir` adds every `.wav` in a
//! directory, sorted by name. The checkpoint is loaded ONCE for the whole
//! batch, which is the difference between seconds and minutes when scoring a
//! sweep: the large-v3 load dominates a single short clip's transcription.
//!
//! This exists to close the TTS evaluation loop: generate speech with
//! `voxcpm_clone`, transcribe it here, and compare the transcript to the text
//! that was synthesised. A model that drops, invents, or slurs words shows up
//! as a word error the moment the two strings differ, which listening alone
//! does not measure repeatably.
//!
//! # Language is not optional in practice
//!
//! Whisper's language token changes what it emits, not merely how it labels
//! the output. On Malay audio with English code-switching, `--language en`
//! makes the reference HuggingFace model emit a single token; `ms` transcribes
//! it. `--language` therefore defaults to `ms` here, matching the corpus this
//! repository is built around. Pass it explicitly for anything else.
//!
//! `--translate` switches the task from transcribe to translate, which emits
//! ENGLISH regardless of the input language — that is a different measurement
//! and is never what a WER check against the synthesised text wants.
//!
//! # Any rate in, 16 kHz to the model
//!
//! Whisper's mel front end is defined at 16 kHz, so the file is decoded and
//! resampled to [`WHISPER_RATE`] here rather than being required to arrive at
//! that rate. Whisper also accepts at most 30 s in one window; longer audio is
//! refused by `transcribe` rather than silently truncated.
//!
//! # Output
//!
//! One `path<TAB>transcript` line per input on stdout, progress on stderr, so
//! `transcribe ... 2>/dev/null` is a TSV a WER script reads directly. A batch
//! keeps going when one file fails: the error goes to stderr and that file has
//! no output row, so one unreadable wav never costs the other sixty-nine.
use std::path::{Path, PathBuf};

use boostr::model::audio::{
    TranscribeOptions, WhisperBundle, decode_audio, extension_hint, to_mono_at_rate,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};

/// Sample rate Whisper's mel front end is defined at.
const WHISPER_RATE: u32 = 16000;

const USAGE: &str = "usage: transcribe --model WHISPER_DIR \
(--audio FILE | --audio-dir DIR) ... [--device cpu|cuda] [--language ms] [--translate]";

/// Runtime the encoder and decoder run on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Cuda,
}

/// Parse a `--device` value.
fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => Err(format!("unknown --device {other}, expected cpu or cuda")),
    }
}

/// Extension `--audio-dir` collects, lowercase.
const WAV_EXTENSION: &str = "wav";

struct Args {
    model: PathBuf,
    /// Every input, in the order the command line named them. `--audio-dir`
    /// contributes its entries sorted by name so a batch is reproducible.
    audio: Vec<PathBuf>,
    language: String,
    translate: bool,
    /// Defaults to `cpu` so a build without CUDA behaves as it always has.
    device: Device,
}

/// Consume the value that follows `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

/// Every `.wav` in `dir`, sorted by name.
///
/// Sorted because an unsorted read gives a different row order per run, and a
/// batch whose output order shifts is a batch whose diff is unreadable.
fn wavs_in(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("--audio-dir {}: {e}", dir.display()))?;
    let mut found = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("--audio-dir {}: {e}", dir.display()))?
            .path();
        let is_wav = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(WAV_EXTENSION));
        if is_wav {
            found.push(path);
        }
    }
    if found.is_empty() {
        return Err(format!("--audio-dir {} holds no .wav", dir.display()));
    }
    found.sort();
    Ok(found)
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut model = None;
    let mut audio: Vec<PathBuf> = Vec::new();
    // See the module docs: `ms` is the corpus this repository transcribes, and
    // the wrong language token changes the OUTPUT, not just its label.
    let mut language = "ms".to_string();
    let mut translate = false;
    let mut device = Device::Cpu;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--model" => model = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audio" => audio.push(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audio-dir" => {
                let dir = PathBuf::from(take_value(&argv, &mut i, flag)?);
                audio.extend(wavs_in(&dir)?);
            }
            "--language" => language = take_value(&argv, &mut i, flag)?,
            "--translate" => translate = true,
            "--device" => device = parse_device(&take_value(&argv, &mut i, flag)?)?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }

    if audio.is_empty() {
        return Err(format!("--audio or --audio-dir is required\n{USAGE}"));
    }
    Ok(Args {
        model: model.ok_or_else(|| format!("--model is required\n{USAGE}"))?,
        audio,
        language,
        translate,
        device,
    })
}

/// Decode `path` and return mono samples at [`WHISPER_RATE`], plus the file's
/// own rate and frame count so the operator can see whether a resample ran.
fn load_audio(path: &Path) -> Result<(Vec<f32>, u32, usize), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint)?;
    let (rate, frames) = (data.sample_rate, data.frames());
    Ok((to_mono_at_rate(&data, WHISPER_RATE)?, rate, frames))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime, _>(&args, &device, &client)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run::<CudaRuntime, _>(&args, &device, &client)
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
}

/// Load the checkpoint once and transcribe every input on runtime `R`.
///
/// The bundle is built before the first file is read, so a batch pays one load
/// no matter how many clips follow it.
fn run<R, C>(args: &Args, device: &R::Device, client: &C) -> Result<(), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + MatmulOps<R>
        + BinaryOps<R>
        + ActivationOps<R>
        + NormalizationOps<R>
        + ConvOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
{
    eprintln!("loading {} ...", args.model.display());
    // Whisper's mel spectrogram is built in F32 and numr's ops require the
    // input and the weight to share a dtype, so an F16 checkpoint (large-v3
    // ships one) fails with `conv1d requires same dtype` unless the weights
    // are cast on the way in. F32 is the only dtype the mel path produces, so
    // it is what every checkpoint is loaded as here.
    let bundle = WhisperBundle::<R>::from_dir_with_dtype(&args.model, device, client, DType::F32)?;

    let options = TranscribeOptions {
        language: Some(&args.language),
        translate: args.translate,
        max_new_tokens: None,
    };

    let mut failed = 0usize;
    for path in &args.audio {
        match transcribe_one(&bundle, client, path, &options) {
            Ok(text) => {
                // The transcript goes to stdout and every progress line to
                // stderr, so `transcribe ... 2>/dev/null` is a TSV a WER
                // script reads directly. A tab inside a transcript would break
                // that column split, so whitespace is normalised to spaces.
                let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
                println!("{}\t{}", path.display(), flat);
            }
            Err(error) => {
                // One bad file must not cost the rest of the batch, so this
                // reports and continues rather than returning.
                failed += 1;
                eprintln!("FAILED {}: {error}", path.display());
            }
        }
    }

    if failed > 0 {
        return Err(format!("{failed} of {} file(s) failed", args.audio.len()).into());
    }
    Ok(())
}

/// Decode one file and transcribe it with an already-loaded `bundle`.
fn transcribe_one<R, C>(
    bundle: &WhisperBundle<R>,
    client: &C,
    path: &Path,
    options: &TranscribeOptions<'_>,
) -> Result<String, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + MatmulOps<R>
        + BinaryOps<R>
        + ActivationOps<R>
        + NormalizationOps<R>
        + ConvOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
{
    let (samples, native_rate, native_frames) = load_audio(path)?;
    eprintln!(
        "audio: {} ({:.2}s, {native_rate} Hz) -> {} samples at {WHISPER_RATE} Hz",
        path.display(),
        native_frames as f64 / native_rate as f64,
        samples.len()
    );
    let out = bundle.transcribe(client, &samples, WHISPER_RATE as usize, options)?;
    Ok(out.text)
}
