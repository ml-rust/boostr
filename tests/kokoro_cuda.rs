//! Kokoro's synthesis path on CUDA, held to the CPU port of the same
//! checkpoint.
//!
//! Run with:
//!   `cd boostr && cargo test --release --features cuda,kokoro --test kokoro_cuda`
//!
//! Every Kokoro struct is generic over `R: Runtime`, so the CUDA run uses the
//! code the CPU run uses. A differing conv padding, a strided-copy error or a
//! reduction-order change still shows up as a moved value, never as a compile
//! error, so this test runs the real checkpoint on both backends:
//!
//! 1. Durations: exact. They are integers decoded from the duration logits.
//! 2. F0, energy, the decoder trunk, the STFT of a fixed excitation, and the
//!    generator over a fixed excitation spectrum: within [`REL_TOL`] of the
//!    CPU value. These stages draw no random number.
//! 3. End to end: two seeded `synthesize` runs on CUDA give identical
//!    samples, all finite, of the length the durations fix. Another seed
//!    gives other samples. Backends run separate PRNGs, so the seeded
//!    waveform is not compared against the CPU one.
//!
//! Skips when the Kokoro model directory is unavailable (`KOKORO_DIR`, else
//! `$BOOSTR_MODELS_DIR/Kokoro-82M`).

#![cfg(feature = "cuda")]

mod common;

use std::f64::consts::PI;
use std::path::Path;

use boostr::format::load_voice_pt;
use boostr::model::audio::kokoro::{
    KokoroClient, KokoroModelV2, KokoroPhonemeVocab, KokoroSynthOptions, length_regulator,
    load_kokoro_v2, select_voice_style, split_voice_style,
};
use common::{model_fixture, skip_notice};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Each value stays within this fraction of the largest CPU value it is
/// checked with: the bar the Kokoro reference parity tests hold the CPU
/// port to.
const REL_TOL: f32 = 1e-4;

/// An excitation phase is compared where its magnitude reaches this fraction
/// of the largest magnitude: below it, f32 rounding moves the phase freely.
const PHASE_FLOOR: f32 = 1e-2;

/// The largest gap, in radians modulo 2π, a well-defined phase keeps.
const PHASE_TOL: f32 = 1e-3;

/// The phonemes and voice the reference parity fixture uses.
const PHONEMES: &str = "həlˈoʊ wˈɜɹld";
const VOICE: &str = "af_heart";

/// The seed of the end-to-end runs.
const SEED: u64 = 42;

/// Every deterministic stage's output, read back to the host.
struct Stages {
    durations: Vec<u32>,
    f0: Vec<f32>,
    n: Vec<f32>,
    trunk: Vec<f32>,
    trunk_width: usize,
    spectrum: Vec<f32>,
    mag: Vec<f32>,
    phase: Vec<f32>,
}

fn values<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<f32> {
    t.contiguous().expect("contiguous").to_vec()
}

/// A 32-bit integer hash of `t`, in `[-0.5, 0.5)`.
fn hiss(t: u64) -> f64 {
    let mut x = t.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF;
    x ^= x >> 16;
    x = x.wrapping_mul(0x45D9F3B) & 0xFFFF_FFFF;
    x ^= x >> 16;
    x as f64 / (1u64 << 32) as f64 - 0.5
}

/// A fixed excitation: sines at 150 and 450 Hz plus `hiss`.
fn source(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            let tones = 0.3 * (2.0 * PI * 150.0 * t / 24000.0).sin()
                + 0.1 * (2.0 * PI * 450.0 * t / 24000.0).sin();
            (tones + 0.2 * hiss(i as u64)) as f32
        })
        .collect()
}

/// A fixed excitation spectrum `[2 * bins, frames]`: smooth magnitudes, then
/// phases inside (-2.5, 2.5).
fn given_spectrum(bins: usize, frames: usize) -> Vec<f32> {
    let row = |b: usize, phase: bool| {
        (0..frames).map(move |f| {
            let (f, b) = (f as f64, b as f64);
            let v = if phase {
                2.5 * (0.003 * f + 0.7 * b).sin()
            } else {
                0.5 + 0.4 * (0.01 * f + b).sin()
            };
            v as f32
        })
    };
    (0..bins)
        .flat_map(|b| row(b, false))
        .chain((0..bins).flat_map(|b| row(b, true)))
        .collect()
}

/// Panics unless `got` and `want` have one length and differ by under
/// [`REL_TOL`] of `want`'s largest magnitude anywhere.
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let tol = REL_TOL * want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let (at, diff) = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .enumerate()
        .fold(
            (0, 0f32),
            |best, (i, d)| if d > best.1 { (i, d) } else { best },
        );
    assert!(
        diff <= tol,
        "{what}: value {at} is {} on CUDA and {} on CPU, off by {diff} (allowed {tol})",
        got[at],
        want[at],
    );
}

/// Panics unless every phase whose magnitude reaches [`PHASE_FLOOR`] of the
/// peak stays within [`PHASE_TOL`] of the CPU phase, modulo 2π.
fn assert_phase_close(got: &[f32], want: &[f32], mag: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = mag.iter().fold(0f32, |m, v| m.max(*v));
    for (i, ((g, w), m)) in got.iter().zip(want).zip(mag).enumerate() {
        if *m < PHASE_FLOOR * peak {
            continue;
        }
        let d = (g - w).rem_euclid(std::f32::consts::TAU);
        let d = d.min(std::f32::consts::TAU - d);
        assert!(
            d <= PHASE_TOL,
            "{what}: phase {i} is {g} on CUDA and {w} on CPU"
        );
    }
}

/// Runs every stage that draws no random number on one backend.
fn stages<R, C>(
    model: &KokoroModelV2<R>,
    client: &C,
    device: &R::Device,
    ids: &[i64],
    voice_pack: &Tensor<R>,
) -> Stages
where
    R: Runtime<DType = DType>,
    C: KokoroClient<R>,
    R::Client: IndexingOps<R>,
{
    let ids = Tensor::<R>::from_slice(ids, &[1, ids.len()], device).expect("ids");
    let row = select_voice_style(voice_pack, PHONEMES.chars().count()).expect("voice row");
    let opts = KokoroSynthOptions {
        speed: 1.0,
        seed: Some(SEED),
    };
    let (_, _, durations) = model
        .forward_to_spectrogram(client, &ids, &row, &opts)
        .expect("spectrogram");

    let (dec_style, pred_style) = split_voice_style(&row, model.config.style_dim).expect("style");
    let d_en = model.bert.forward(client, &ids).expect("bert");
    let (_, features) = model
        .predictor
        .predict_duration_features(client, &d_en, &pred_style)
        .expect("duration features");
    let frames = length_regulator(client, &features, &durations).expect("regulate");
    let (f0, n) = model
        .predictor
        .predict_f0_n(client, &frames, &pred_style)
        .expect("F0 and energy");
    let t_en = model
        .text_encoder
        .forward(client, &ids)
        .expect("text encoder");
    let asr = length_regulator(client, &t_en, &durations)
        .expect("regulate")
        .transpose(1, 2)
        .expect("transpose")
        .contiguous()
        .expect("contiguous");
    let trunk = model
        .decoder
        .trunk(client, &asr, &f0, &n, &dec_style)
        .expect("decoder trunk");
    let trunk_width = trunk.shape()[2];

    let generator = &model.decoder.generator;
    let source_len = trunk_width * generator.f0_upsample_factor * generator.stft.hop_length;
    let src =
        Tensor::<R>::from_slice(&source(source_len), &[1, source_len], device).expect("source");
    let spectrum = generator
        .source_spectrum(client, &src)
        .expect("excitation spectrum");

    let bins = generator.stft.n_fft / 2 + 1;
    // One STFT frame per hop of the source waveform, plus the closing frame.
    let spec_frames = trunk_width * generator.f0_upsample_factor / generator.stft.hop_length + 1;
    let given = Tensor::<R>::from_slice(
        &given_spectrum(bins, spec_frames),
        &[1, 2 * bins, spec_frames],
        device,
    )
    .expect("given spectrum");
    let (mag, phase) = generator
        .forward_with_spectrum(client, &trunk, &dec_style, &given)
        .expect("generator");

    Stages {
        durations,
        f0: values(&f0),
        n: values(&n),
        trunk: values(&trunk),
        trunk_width,
        spectrum: values(&spectrum),
        mag: values(&mag),
        phase: values(&phase),
    }
}

/// The checkpoint's ids for [`PHONEMES`]: one per character, bracketed by
/// the reserved id 0.
fn phoneme_ids(dir: &Path) -> Vec<i64> {
    let vocab = KokoroPhonemeVocab::from_json_file(dir.join("config.json")).expect("vocab");
    let tokens: Vec<String> = PHONEMES.chars().map(String::from).collect();
    let inner = vocab
        .encode_strict(&tokens)
        .expect("every phoneme is in the vocab");
    std::iter::once(0)
        .chain(inner.into_iter().map(i64::from))
        .chain(std::iter::once(0))
        .collect()
}

#[test]
fn cuda_matches_cpu_and_a_seed_reproduces_the_waveform() {
    let Some(dir) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M model directory", "KOKORO_DIR");
        return;
    };
    let voice_path = dir.join(format!("voices/{VOICE}.pt"));
    if !voice_path.exists() {
        eprintln!("skipping: {} is absent", voice_path.display());
        return;
    }
    let ids = phoneme_ids(&dir);

    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let cpu_model = load_kokoro_v2::<CpuRuntime, _>(&cpu_client, &dir, &cpu_device)
        .expect("loading Kokoro on CPU");
    let cpu_pack = load_voice_pt::<CpuRuntime>(&voice_path, &cpu_device).expect("CPU voice");

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let model =
        load_kokoro_v2::<CudaRuntime, _>(&client, &dir, &device).expect("loading Kokoro on CUDA");
    let pack = load_voice_pt::<CudaRuntime>(&voice_path, &device).expect("CUDA voice");

    let want = stages(&cpu_model, &cpu_client, &cpu_device, &ids, &cpu_pack);
    let got = stages(&model, &client, &device, &ids, &pack);

    assert_eq!(got.durations, want.durations, "durations");
    assert_close(&got.f0, &want.f0, "F0");
    assert_close(&got.n, &want.n, "energy");
    assert_eq!(got.trunk_width, want.trunk_width, "trunk width");
    assert_close(&got.trunk, &want.trunk, "decoder trunk");

    let half = want.spectrum.len() / 2;
    let (want_mag, want_phase) = want.spectrum.split_at(half);
    let (got_mag, got_phase) = got.spectrum.split_at(half);
    assert_close(got_mag, want_mag, "excitation magnitude");
    assert_phase_close(got_phase, want_phase, want_mag, "excitation");

    assert_close(&got.mag, &want.mag, "generator magnitude");
    assert_close(&got.phase, &want.phase, "generator phase");

    let ids_t = Tensor::<CudaRuntime>::from_slice(&ids, &[1, ids.len()], &device).expect("ids");
    let row = select_voice_style(&pack, PHONEMES.chars().count()).expect("voice row");
    let synth = |seed| {
        let opts = KokoroSynthOptions { speed: 1.0, seed };
        let wave = model
            .synthesize(&client, &ids_t, &row, &opts)
            .expect("synthesis");
        values(&wave)
    };
    let first = synth(Some(SEED));
    let second = synth(Some(SEED));
    let generator = &model.decoder.generator;
    // One sample per upsampled F0 frame: f0_upsample_factor is samples per trunk frame.
    let expected_len = want.trunk_width * generator.f0_upsample_factor;
    assert_eq!(first.len(), expected_len, "samples");
    assert!(
        first.iter().all(|s| s.is_finite()),
        "the seeded waveform holds a non-finite sample"
    );
    let peak = first.iter().fold(0f32, |m, s| m.max(s.abs()));
    let worst = first
        .iter()
        .zip(&second)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let diffs = first.iter().zip(&second).filter(|(a, b)| a != b).count();
    assert_eq!(
        first, second,
        "one seed must reproduce the waveform exactly: {diffs} samples differ, worst {worst} against peak {peak}"
    );
    assert_ne!(
        first,
        synth(Some(SEED + 1)),
        "another seed must change the source draws"
    );
}
