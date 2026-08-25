//! End-to-end Whisper: real checkpoint off disk, real mel, real greedy decode.
//!
//! Nothing else in `tests/` loads a Whisper checkpoint, so `WhisperBundle::from_dir`
//! and the tensor-name plumbing in `WhisperModel::from_varbuilder` are only
//! exercised here. Every test is fixture-gated and skips when the fixture is
//! absent, so the default suite stays green on a machine without the models.
//!
//! Fixtures (all under `$BOOSTR_MODELS_DIR`, each overridable by its own var):
//! * `WHISPER_TINY_DIR` → `whisper-tiny` — HF layout, 80 mel bins, vocab 51865
//! * `WHISPER_LARGE_V3_DIR` → `whisper-large-v3` — 128 mel bins, vocab 51866, 3 GB
//! * `WHISPER_MEL_FIXTURE` → `whisper_mel_fixture.safetensors` — `input` `[480000]`,
//!   `mel_80` `[80, 3000]`, `mel_128` `[128, 3000]`
//! * `whisper-tiny_reference.json` / `whisper-large-v3_reference.json` — the greedy
//!   transcription HuggingFace's own `WhisperForConditionalGeneration` produces for
//!   that same `input`, language `ms`, task transcribe
//!
//! The audio is Malay with English technical code-switching, hence language `"ms"`.
//! With `"en"` the reference model emits a single token — do not "correct" this.

use std::path::{Path, PathBuf};

use boostr::model::audio::{GenerateOptions, SpeechSegment, TranscribeOptions, WhisperBundle};
use boostr::nn::VarMap;
#[cfg(feature = "f16")]
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use serde::Deserialize;

mod common;
use common::{cpu_setup, model_fixture, skip_notice};

const SAMPLE_RATE: usize = 16000;
const FRAMES: usize = 3000;

/// The reference transcription captured from upstream HuggingFace.
#[derive(Debug, Deserialize)]
struct Reference {
    #[allow(dead_code)]
    checkpoint: String,
    language: String,
    #[allow(dead_code)]
    sample_rate: usize,
    #[allow(dead_code)]
    num_samples: usize,
    /// The generated ids ONLY — the SOT prompt prefix is not included, and
    /// neither is the trailing `<|endoftext|>`.
    token_ids: Vec<u32>,
    text: String,
    /// Fixed sub-ranges transcribed independently, for `transcribe_segments`.
    /// Their bounds must match the segments the test builds.
    segments: Vec<ReferenceSegment>,
}

#[derive(Debug, Deserialize)]
struct ReferenceSegment {
    start: usize,
    end: usize,
    token_ids: Vec<u32>,
    text: String,
}

fn load_reference(path: &Path) -> Reference {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

fn mel_fixture(device: &CpuDevice) -> Option<VarMap<CpuRuntime>> {
    let path = model_fixture("WHISPER_MEL_FIXTURE", "whisper_mel_fixture.safetensors")?;
    match VarMap::<CpuRuntime>::from_safetensors(&path, device) {
        Ok(map) => Some(map),
        Err(e) => panic!("failed to load {}: {e}", path.display()),
    }
}

fn fixture_tensor(map: &VarMap<CpuRuntime>, name: &str) -> Vec<f32> {
    map.get_tensor(name)
        .unwrap_or_else(|e| panic!("fixture is missing `{name}`: {e}"))
        .to_vec()
}

/// Everything the tiny-checkpoint tests need, or `None` when any part is absent.
struct TinyFixtures {
    dir: PathBuf,
    reference: PathBuf,
    client: CpuClient,
    device: CpuDevice,
}

fn tiny_fixtures() -> Option<TinyFixtures> {
    let dir = model_fixture("WHISPER_TINY_DIR", "whisper-tiny")?;
    let reference = model_fixture("WHISPER_TINY_REFERENCE", "whisper-tiny_reference.json")?;
    let (client, device) = cpu_setup();
    Some(TinyFixtures {
        dir,
        reference,
        client,
        device,
    })
}

/// Greedy options taken from the CHECKPOINT'S OWN `generation_config.json`,
/// which is what upstream applies: the `suppress_tokens` list, the
/// `begin_suppress_tokens` list at the first generated position only, and the
/// configured eos.
///
/// Building these by hand instead is how this test first failed — an unmasked
/// token 503 (`' "'`) is in tiny's suppress list, and picking it sent the
/// decode into a repetition loop.
///
/// `budget` bounds the run so a divergent decode cannot spin for 448 steps.
fn greedy_options(bundle: &WhisperBundle<CpuRuntime>, budget: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: budget,
        ..bundle.generate_options()
    }
}

/// Compare two id sequences, printing both in full first so a divergence is
/// readable rather than a bare `assert_eq` dump.
fn assert_ids_match(what: &str, produced: &[u32], reference: &[u32]) {
    eprintln!("{what}: produced  ({}) {produced:?}", produced.len());
    eprintln!("{what}: reference ({}) {reference:?}", reference.len());
    if let Some(i) = produced
        .iter()
        .zip(reference.iter())
        .position(|(a, b)| a != b)
    {
        eprintln!(
            "{what}: first divergence at index {i}: produced {} vs reference {}",
            produced[i], reference[i]
        );
    }
    assert_eq!(produced, reference, "{what}: greedy ids differ");
}

#[test]
fn whisper_tiny_bundle_loads() {
    let Some(fx) = tiny_fixtures() else {
        skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        return;
    };
    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir(&fx.dir, &fx.device).expect("load whisper-tiny");

    assert_eq!(bundle.num_mel_bins, 80, "tiny is an 80-bin checkpoint");
    assert_eq!(bundle.config.hidden_size, 384, "tiny d_model");
    assert_eq!(bundle.config.vocab_size, 51865, "tiny vocab_size");

    // The SOT prompt is what every decode starts from; a wrong variant detection
    // shows up here as wrong ids rather than as a confusing decode failure.
    let prompt = bundle.sot_prompt(Some("ms"), false);
    eprintln!("tiny sot_prompt(ms, transcribe) = {prompt:?}");
    assert_eq!(
        prompt.len(),
        4,
        "multilingual prompt is [sot, lang, task, notimestamps]"
    );
}

#[test]
fn whisper_tiny_encodes_reference_mel() {
    let Some(fx) = tiny_fixtures() else {
        skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        return;
    };
    let Some(map) = mel_fixture(&fx.device) else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir(&fx.dir, &fx.device).expect("load whisper-tiny");

    let mel = fixture_tensor(&map, "mel_80");
    assert_eq!(mel.len(), 80 * FRAMES, "mel_80 is not [80, 3000]");
    let mel_t =
        Tensor::<CpuRuntime>::from_slice(&mel, &[1, 80, FRAMES], &fx.device).expect("mel tensor");

    let out = bundle
        .model
        .encode(&fx.client, &mel_t)
        .expect("whisper encode");

    eprintln!("tiny encoder output shape = {:?}", out.shape());
    // Two conv layers with stride 1 then 2 halve 3000 frames to 1500 positions,
    // each of width d_model.
    assert_eq!(
        out.shape(),
        &[1, 1500, 384],
        "encoder output shape does not match the config"
    );

    let data: Vec<f32> = out.to_vec();
    assert!(
        data.iter().all(|x| x.is_finite()),
        "encoder output contains NaN or infinity"
    );
}

#[test]
fn whisper_tiny_greedy_matches_reference() {
    let Some(fx) = tiny_fixtures() else {
        skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        return;
    };
    let Some(map) = mel_fixture(&fx.device) else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    let reference = load_reference(&fx.reference);
    assert_eq!(reference.language, "ms", "reference must be the Malay run");

    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir(&fx.dir, &fx.device).expect("load whisper-tiny");

    // The REFERENCE mel, not ours: this test isolates the model from the front
    // end, so a mel regression cannot be mistaken for a decoder regression.
    let mel = fixture_tensor(&map, "mel_80");
    let mel_t =
        Tensor::<CpuRuntime>::from_slice(&mel, &[1, 80, FRAMES], &fx.device).expect("mel tensor");
    let encoded = bundle
        .model
        .encode(&fx.client, &mel_t)
        .expect("whisper encode");

    let prompt = bundle.sot_prompt(Some(&reference.language), false);
    // `WhisperModel::generate` returns the GENERATED tokens only — it never
    // echoes `start_tokens` — and stops before emitting an eos id. The reference
    // ids are likewise prompt-free and eos-free, so neither side needs trimming.
    let ids = bundle
        .model
        .generate(
            &fx.client,
            &encoded,
            &prompt,
            &greedy_options(&bundle, reference.token_ids.len() + 16),
        )
        .expect("whisper generate");

    assert_ids_match("whisper-tiny", &ids, &reference.token_ids);
}

#[test]
fn whisper_tiny_end_to_end_from_samples() {
    let Some(fx) = tiny_fixtures() else {
        skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        return;
    };
    let Some(map) = mel_fixture(&fx.device) else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    let reference = load_reference(&fx.reference);
    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir(&fx.dir, &fx.device).expect("load whisper-tiny");

    // Waveform → mel front end → encoder → greedy → text, all inside
    // `transcribe`. This is the only test that puts the mel front end and the
    // model on the same path, so it is what proves the two agree — and it now
    // also proves `transcribe` assembles that path exactly as a caller would.
    let samples = fixture_tensor(&map, "input");
    let out = bundle
        .transcribe(
            &fx.client,
            &samples,
            SAMPLE_RATE,
            &TranscribeOptions {
                language: Some(&reference.language),
                translate: false,
                max_new_tokens: Some(reference.token_ids.len() + 16),
            },
        )
        .expect("whisper transcribe");

    eprintln!("end-to-end produced : {:?}", out.text);
    eprintln!("end-to-end reference: {:?}", reference.text);
    assert_eq!(
        out.text, reference.text,
        "end-to-end transcription differs from the HuggingFace reference"
    );
}

/// Over-long audio must be REFUSED, never silently trimmed to the first 30 s.
/// The fixture `input` is exactly 30 s, so two copies are exactly 60 s.
#[test]
fn whisper_transcribe_rejects_audio_over_thirty_seconds() {
    let Some(fx) = tiny_fixtures() else {
        skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        return;
    };
    let Some(map) = mel_fixture(&fx.device) else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir(&fx.dir, &fx.device).expect("load whisper-tiny");

    let samples = fixture_tensor(&map, "input");
    assert_eq!(samples.len(), 30 * SAMPLE_RATE, "fixture input is not 30 s");
    let doubled: Vec<f32> = samples.iter().chain(samples.iter()).copied().collect();

    let err = bundle
        .transcribe(
            &fx.client,
            &doubled,
            SAMPLE_RATE,
            &TranscribeOptions::default(),
        )
        .expect_err("60 s of audio must be refused, not truncated");
    let msg = err.to_string();
    eprintln!("over-long rejection: {msg}");
    assert!(
        msg.contains("60.000"),
        "the error must name the actual duration, got: {msg}"
    );
    assert!(
        msg.contains("30 s window"),
        "the error must name the 30 s limit, got: {msg}"
    );
}

/// `transcribe_segments` drives one decode per segment, and refuses a segment
/// that runs past the end of the buffer instead of panicking on the slice.
#[test]
fn whisper_transcribe_segments_covers_each_range() {
    let Some(fx) = tiny_fixtures() else {
        skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        return;
    };
    let Some(map) = mel_fixture(&fx.device) else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir(&fx.dir, &fx.device).expect("load whisper-tiny");

    let samples = fixture_tensor(&map, "input");
    let reference = load_reference(&fx.reference);
    let segments: Vec<SpeechSegment> = reference
        .segments
        .iter()
        .map(|s| SpeechSegment {
            start: s.start,
            end: s.end,
        })
        .collect();
    assert!(!segments.is_empty(), "reference carries no segments");

    let opts = TranscribeOptions {
        language: Some(reference.language.as_str()),
        translate: false,
        // 48 is what the reference was captured with, and it must match
        // exactly: segment 1 HITS this cap — whisper-tiny falls into a
        // repetition loop on that range — so any other budget diverges. See
        // `audio/pipeline/make_whisper_asr_reference.py`.
        max_new_tokens: Some(48),
    };

    let out = bundle
        .transcribe_segments(&fx.client, &samples, SAMPLE_RATE, &segments, &opts)
        .expect("transcribe segments");
    assert_eq!(
        out.len(),
        reference.segments.len(),
        "one transcription per segment"
    );
    // Exact equality, not a non-empty check: HuggingFace produces these
    // byte-for-byte on the same ranges. Segment 0 comes back in ENGLISH even
    // though the audio is Malay and the prompt carries `<|ms|>` + transcribe —
    // whisper-tiny translates instead of transcribing, and upstream does the
    // same, so matching it is correct behaviour, not a bug to chase.
    for (i, (got, want)) in out.iter().zip(reference.segments.iter()).enumerate() {
        eprintln!("segment {i} ours: {:?}", got.text);
        eprintln!("segment {i} ref : {:?}", want.text);
        assert_eq!(got.tokens, want.token_ids, "segment {i}: token ids differ");
        assert_eq!(got.text, want.text, "segment {i}: text differs");
    }

    let past_end = [SpeechSegment {
        start: 0,
        end: samples.len() + 1,
    }];
    let err = bundle
        .transcribe_segments(&fx.client, &samples, SAMPLE_RATE, &past_end, &opts)
        .expect_err("a segment past the end of the buffer must be an error");
    let msg = err.to_string();
    eprintln!("out-of-bounds rejection: {msg}");
    assert!(
        msg.contains("segment 0"),
        "the error must name the offending index, got: {msg}"
    );
}

/// large-v3 is a 3 GB load and a 32+32-layer decode on CPU (~5 minutes), so it
/// is `#[ignore]`d on top of being fixture-gated. Run it with:
/// `cargo nextest run --features f16 --test whisper_integration --run-ignored all`
///
/// The `f16` gate is required, not incidental: large-v3 stores fp16 weights and
/// numr's CPU `cast` rejects F16 without that feature
/// (`UnsupportedDType { dtype: F16, op: "cast" }`).
#[cfg(feature = "f16")]
#[test]
#[ignore]
fn whisper_large_v3_greedy_matches_reference() {
    let Some(dir) = model_fixture("WHISPER_LARGE_V3_DIR", "whisper-large-v3") else {
        skip_notice("whisper-large-v3 checkpoint", "WHISPER_LARGE_V3_DIR");
        return;
    };
    let Some(ref_path) = model_fixture(
        "WHISPER_LARGE_V3_REFERENCE",
        "whisper-large-v3_reference.json",
    ) else {
        skip_notice("whisper-large-v3 reference", "WHISPER_LARGE_V3_REFERENCE");
        return;
    };
    let (client, device) = cpu_setup();
    let Some(map) = mel_fixture(&device) else {
        skip_notice("whisper mel fixture", "WHISPER_MEL_FIXTURE");
        return;
    };
    let reference = load_reference(&ref_path);

    // large-v3 ships fp16 weights, so it must be cast on load: numr requires the
    // input and the weight to share a dtype, and the mel is f32.
    let bundle =
        WhisperBundle::<CpuRuntime>::from_dir_with_dtype(&dir, &device, &client, DType::F32)
            .expect("load whisper-large-v3");
    assert_eq!(bundle.num_mel_bins, 128, "large-v3 is a 128-bin checkpoint");
    assert_eq!(bundle.config.hidden_size, 1280, "large-v3 d_model");
    assert_eq!(bundle.config.vocab_size, 51866, "large-v3 vocab_size");

    let mel = fixture_tensor(&map, "mel_128");
    assert_eq!(mel.len(), 128 * FRAMES, "mel_128 is not [128, 3000]");
    let mel_t =
        Tensor::<CpuRuntime>::from_slice(&mel, &[1, 128, FRAMES], &device).expect("mel tensor");
    let encoded = bundle
        .model
        .encode(&client, &mel_t)
        .expect("whisper encode");
    assert_eq!(encoded.shape(), &[1, 1500, 1280], "encoder output shape");

    let prompt = bundle.sot_prompt(Some(&reference.language), false);
    let ids = bundle
        .model
        .generate(
            &client,
            &encoded,
            &prompt,
            &greedy_options(&bundle, reference.token_ids.len() + 16),
        )
        .expect("whisper generate");

    assert_ids_match("whisper-large-v3", &ids, &reference.token_ids);
}
