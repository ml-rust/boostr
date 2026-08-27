//! Transcribe an audio file with Whisper, in pure Rust.
//!
//! ```text
//! cargo run --release --features audio --example transcribe -- \
//!     --model WHISPER_DIR --audio FILE [--language ms] [--translate]
//! ```
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
use std::path::{Path, PathBuf};

use boostr::model::audio::{
    TranscribeOptions, WhisperBundle, decode_audio, extension_hint, to_mono_at_rate,
};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

/// Sample rate Whisper's mel front end is defined at.
const WHISPER_RATE: u32 = 16000;

const USAGE: &str = "usage: transcribe --model WHISPER_DIR --audio FILE \
[--language ms] [--translate]";

struct Args {
    model: PathBuf,
    audio: PathBuf,
    language: String,
    translate: bool,
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
    let (mut model, mut audio) = (None, None);
    // See the module docs: `ms` is the corpus this repository transcribes, and
    // the wrong language token changes the OUTPUT, not just its label.
    let mut language = "ms".to_string();
    let mut translate = false;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--model" => model = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audio" => audio = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--language" => language = take_value(&argv, &mut i, flag)?,
            "--translate" => translate = true,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }

    Ok(Args {
        model: model.ok_or_else(|| format!("--model is required\n{USAGE}"))?,
        audio: audio.ok_or_else(|| format!("--audio is required\n{USAGE}"))?,
        language,
        translate,
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

    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let (samples, native_rate, native_frames) = load_audio(&args.audio)?;
    eprintln!(
        "audio: {} ({:.2}s, {native_rate} Hz) -> {} samples at {WHISPER_RATE} Hz",
        args.audio.display(),
        native_frames as f64 / native_rate as f64,
        samples.len()
    );

    eprintln!("loading {} ...", args.model.display());
    // Whisper's mel spectrogram is built in F32 and numr's ops require the
    // input and the weight to share a dtype, so an F16 checkpoint (large-v3
    // ships one) fails with `conv1d requires same dtype` unless the weights
    // are cast on the way in. F32 is the only dtype the mel path produces, so
    // it is what every checkpoint is loaded as here.
    let bundle = WhisperBundle::<CpuRuntime>::from_dir_with_dtype(
        &args.model,
        &device,
        &client,
        DType::F32,
    )?;

    let out = bundle.transcribe(
        &client,
        &samples,
        WHISPER_RATE as usize,
        &TranscribeOptions {
            language: Some(&args.language),
            translate: args.translate,
            max_new_tokens: None,
        },
    )?;

    // The transcript goes to stdout and every progress line to stderr, so
    // `transcribe ... 2>/dev/null` is pipeable into a WER script.
    println!("{}", out.text);
    Ok(())
}
