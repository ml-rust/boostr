//! Numerical parity for the mel front end against HuggingFace's
//! `WhisperFeatureExtractor`.
//!
//! The fixture is a safetensors file produced by running upstream's own
//! extractor over 30 s of real speech at 16 kHz. It is NOT checked in (5 MB of
//! audio and reference features), so these tests skip unless
//! `WHISPER_MEL_FIXTURE` points at it, or `BOOSTR_MODELS_DIR` contains
//! `whisper_mel_fixture.safetensors`.
//!
//! Tensors, all f32:
//! * `input` `[480000]` — the waveform
//! * `mel_80` `[80, 3000]`, `mel_128` `[128, 3000]` — reference features
//! * `input_short` `[8000]` — 0.5 s of the same audio
//! * `mel_80_short` `[80, 3000]` — reference for `input_short`
//!
//! What this pins that unit tests cannot: the Slaney mel warping and Slaney
//! filter normalization, the reflect-pad-then-drop-last-frame framing, and
//! Whisper's GLOBAL log floor. Every one of those is a silent behavior
//! difference that produces a plausible-looking spectrogram.

use boostr::model::audio::{MelOptions, compute_mel_spectrogram_with};
use boostr::nn::VarMap;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

mod common;
use common::{model_fixture, skip_notice};

const SAMPLE_RATE: usize = 16000;
const FRAMES: usize = 3000;
/// Max absolute difference allowed against the reference, in Whisper's
/// output units (log10 energy, floored 8 below the global max, then
/// `(x + 4) / 4`).
///
/// The three cases actually land at 9.4e-6, 1.5e-5 and 1.6e-5 — f32 rounding,
/// nothing structural. This bound keeps roughly 6x headroom for FFT ordering
/// differences across machines. Do NOT relax it to paper over a real
/// divergence: every one of the six ways this front end can differ from the
/// reference (n_fft, mel scale, filter normalisation, log compression,
/// centering, pad-to-30s) moves the diff by 1e-2 or more, so anything above
/// 1e-4 means a formula is wrong, not that the tolerance is tight.
const TOL: f32 = 1e-4;

/// Largest absolute difference, plus the index where it occurs.
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// Load the fixture, or `None` when it is absent (the test then skips).
fn fixture() -> Option<VarMap<CpuRuntime>> {
    let path = model_fixture("WHISPER_MEL_FIXTURE", "whisper_mel_fixture.safetensors")?;
    let device = CpuDevice::new();
    match VarMap::<CpuRuntime>::from_safetensors(&path, &device) {
        Ok(map) => Some(map),
        Err(e) => panic!("failed to load {}: {e}", path.display()),
    }
}

fn tensor(map: &VarMap<CpuRuntime>, name: &str) -> Vec<f32> {
    map.get_tensor(name)
        .unwrap_or_else(|e| panic!("fixture is missing `{name}`: {e}"))
        .to_vec()
}

/// Run one case end to end and compare against the reference.
fn check(map: &VarMap<CpuRuntime>, input_name: &str, ref_name: &str, num_mel_bins: usize) {
    let samples = tensor(map, input_name);
    let reference = tensor(map, ref_name);
    assert_eq!(
        reference.len(),
        num_mel_bins * FRAMES,
        "reference `{ref_name}` is not [{num_mel_bins}, {FRAMES}]"
    );

    let opts = MelOptions::whisper(num_mel_bins, SAMPLE_RATE);
    let mel = compute_mel_spectrogram_with(&samples, SAMPLE_RATE, &opts).expect("mel");
    assert_eq!(
        mel.len(),
        num_mel_bins * FRAMES,
        "output shape is not [{num_mel_bins}, {FRAMES}]"
    );

    let (worst, at) = max_abs_diff(&mel, &reference);
    let (bin, frame) = (at / FRAMES, at % FRAMES);
    eprintln!(
        "{ref_name}: max abs diff {worst:e} at bin {bin} frame {frame} \
         (ours {}, reference {})",
        mel[at], reference[at]
    );
    assert!(worst < TOL, "{ref_name}: max abs diff {worst:e} >= {TOL:e}");
}

#[test]
fn whisper_mel_80_matches_reference() {
    let Some(map) = fixture() else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    check(&map, "input", "mel_80", 80);
}

#[test]
fn whisper_mel_128_matches_reference() {
    let Some(map) = fixture() else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    // large-v3's config. Same audio, so a bin-count-dependent filterbank bug
    // that happens to cancel at 80 bins still shows up here.
    check(&map, "input", "mel_128", 128);
}

#[test]
fn whisper_mel_short_input_matches_reference() {
    let Some(map) = fixture() else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    // THIS is the case that pins the reflect-padding and the pad-to-30 s step.
    // The input is 0.5 s yet the reference is still [80, 3000]: the signal is
    // zero-filled out to 480000 samples BEFORE the reflect pad, so the mirror
    // at each end reflects the zero-filled buffer, not the 8000 real samples.
    // Skipping either step still yields a well-formed spectrogram — just the
    // wrong one.
    check(&map, "input_short", "mel_80_short", 80);
}
