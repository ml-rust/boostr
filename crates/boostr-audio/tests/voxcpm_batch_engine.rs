//! Concurrent requests to one [`VoxCpm2Engine`] batch into one generation
//! and still render exactly what each request renders alone.
//!
//! The engine queues every request and its owner thread groups whatever is
//! queued at the same time. This test issues several requests at once, then
//! issues the same requests one at a time, and asserts the waveforms match
//! bit for bit — a row's noise draw depends on neither the batch width nor
//! its neighbours, so batching must not change a single sample.
//!
//! Gated on a real model: set `VOXCPM2_GGUF` to a compressr GGUF that embeds
//! the VAE and the tokenizer (the test skips when it is unset or the path is
//! absent). `VOXCPM2_VOICES_DIR`, optional, points at a directory of
//! reference recordings; when set, one request renders with the first voice
//! it lists as well as zero-shot. With the `cuda` feature the engine runs on
//! device 0 with an F16 decoder, the configuration blazr serves; without it,
//! on the CPU at F32.

#![cfg(feature = "voxcpm")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use boostr::model::audio::voxcpm::VoxCpm2Weights;
use boostr_audio::TtsEngine;
use boostr_audio::voxcpm::{
    VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions, ZERO_SHOT_VOICE_ID,
};

/// Three prompts of different length, so the batch is left-padded and the
/// rows finish at different steps.
const PROMPTS: [&str; 3] = [
    "Selamat pagi, today we test batched synthesis.",
    "The quick brown fox jumps over the lazy dog, dan kemudian ia tidur di bawah pokok.",
    "Satu dua tiga.",
];

/// Long enough that every concurrent request lands in the same drain window
/// whatever the machine is doing.
const WINDOW: Duration = Duration::from_millis(500);

fn gguf_fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_GGUF").ok()?);
    path.is_file().then_some(path)
}

fn voices_fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_VOICES_DIR").ok()?);
    path.is_dir().then_some(path)
}

fn load_engine(gguf: &Path, voices: Option<&Path>) -> Arc<dyn TtsEngine> {
    let weights = VoxCpm2Weights::Gguf {
        path: gguf.to_path_buf(),
        config: None,
    };
    let synth = VoxCpm2SynthOptions {
        batch_window: WINDOW,
        max_batch: 8,
        ..VoxCpm2SynthOptions::default()
    };
    #[cfg(feature = "cuda")]
    {
        use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
        use numr::runtime::Runtime;
        let device = CudaDevice::new(0);
        let client = Arc::new(CudaRuntime::default_client(&device));
        let options = VoxCpm2LoadOptions {
            dtype: None,
            vae_decoder_dtype: Some(numr::dtype::DType::F16),
            synth,
            adapter: None,
        };
        Arc::new(
            VoxCpm2Engine::<CudaRuntime>::load(&weights, None, voices, &device, client, options)
                .expect("load VoxCPM2 on CUDA"),
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        use numr::runtime::Runtime;
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let client = Arc::new(CpuRuntime::default_client(&device));
        let options = VoxCpm2LoadOptions {
            dtype: None,
            vae_decoder_dtype: None,
            synth,
            adapter: None,
        };
        Arc::new(
            VoxCpm2Engine::<CpuRuntime>::load(&weights, None, voices, &device, client, options)
                .expect("load VoxCPM2 on CPU"),
        )
    }
}

fn assert_bit_identical(batched: &[f32], alone: &[f32], label: &str) {
    assert_eq!(
        batched.len(),
        alone.len(),
        "{label}: batched {} samples, alone {}",
        batched.len(),
        alone.len()
    );
    let mismatch = batched
        .iter()
        .zip(alone)
        .position(|(b, a)| b.to_bits() != a.to_bits());
    if let Some(i) = mismatch {
        panic!(
            "{label}: sample {i} is {:e} ({:#010x}) batched, {:e} ({:#010x}) alone",
            batched[i],
            batched[i].to_bits(),
            alone[i],
            alone[i].to_bits()
        );
    }
}

/// The request set: every prompt zero-shot, plus the first prompt with the
/// first listed voice when a voices directory is set.
fn requests(engine: &dyn TtsEngine) -> Vec<(String, String)> {
    let mut requests: Vec<(String, String)> = PROMPTS
        .iter()
        .map(|text| ((*text).to_string(), ZERO_SHOT_VOICE_ID.to_string()))
        .collect();
    if let Some(voice) = engine
        .voices()
        .into_iter()
        .find(|voice| voice.id != ZERO_SHOT_VOICE_ID)
    {
        requests.push((PROMPTS[0].to_string(), voice.id));
    }
    requests
}

/// Several `synthesize` calls at once render what each renders alone.
#[test]
fn concurrent_buffered_requests_match_single_renders() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let voices = voices_fixture();
    let engine = load_engine(&gguf, voices.as_deref());
    let requests = requests(engine.as_ref());

    let batched: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = requests
            .iter()
            .map(|(text, voice)| {
                let engine = Arc::clone(&engine);
                scope.spawn(move || {
                    engine
                        .synthesize(text, voice, 1.0)
                        .expect("concurrent synthesize")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("render thread"))
            .collect()
    });

    for (i, (text, voice)) in requests.iter().enumerate() {
        let alone = engine.synthesize(text, voice, 1.0).expect("synthesize");
        assert!(
            alone.iter().all(|v| v.is_finite()),
            "request {i} has non-finite samples"
        );
        assert_bit_identical(&batched[i], &alone, &format!("request {i} ({voice})"));
    }
}

/// A streaming request batched beside buffered ones: its chunks concatenate
/// to the same waveform, and the buffered rows are unaffected.
#[test]
fn a_streaming_request_batches_with_buffered_ones() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let voices = voices_fixture();
    let engine = load_engine(&gguf, voices.as_deref());

    let (streamed, buffered) = std::thread::scope(|scope| {
        let streaming = {
            let engine = Arc::clone(&engine);
            scope.spawn(move || {
                let mut chunks: Vec<Vec<f32>> = Vec::new();
                engine
                    .synthesize_stream(PROMPTS[0], ZERO_SHOT_VOICE_ID, 1.0, &mut |samples| {
                        chunks.push(samples.to_vec());
                        Ok(())
                    })
                    .expect("synthesize_stream");
                chunks.concat()
            })
        };
        let others: Vec<_> = PROMPTS[1..]
            .iter()
            .map(|text| {
                let engine = Arc::clone(&engine);
                scope.spawn(move || {
                    engine
                        .synthesize(text, ZERO_SHOT_VOICE_ID, 1.0)
                        .expect("synthesize")
                })
            })
            .collect();
        (
            streaming.join().expect("streaming thread"),
            others
                .into_iter()
                .map(|handle| handle.join().expect("render thread"))
                .collect::<Vec<_>>(),
        )
    });

    let alone = engine
        .synthesize(PROMPTS[0], ZERO_SHOT_VOICE_ID, 1.0)
        .expect("synthesize");
    assert_bit_identical(&streamed, &alone, "streamed row");
    for (i, text) in PROMPTS[1..].iter().enumerate() {
        let alone = engine
            .synthesize(text, ZERO_SHOT_VOICE_ID, 1.0)
            .expect("synthesize");
        assert_bit_identical(&buffered[i], &alone, &format!("buffered row {i}"));
    }
}

/// A sink that fails mid-utterance cancels its own row and leaves the rows
/// batched beside it intact.
#[test]
fn a_cancelled_streaming_row_does_not_disturb_the_batch() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let voices = voices_fixture();
    let engine = load_engine(&gguf, voices.as_deref());

    let (cancelled, kept) = std::thread::scope(|scope| {
        let cancelling = {
            let engine = Arc::clone(&engine);
            scope.spawn(move || {
                let mut seen = 0usize;
                let result =
                    engine.synthesize_stream(PROMPTS[1], ZERO_SHOT_VOICE_ID, 1.0, &mut |_| {
                        seen += 1;
                        if seen >= 2 {
                            return Err(boostr_audio::Error::DataError {
                                reason: "client went away".into(),
                            });
                        }
                        Ok(())
                    });
                (seen, result)
            })
        };
        let keeping = {
            let engine = Arc::clone(&engine);
            scope.spawn(move || {
                engine
                    .synthesize(PROMPTS[0], ZERO_SHOT_VOICE_ID, 1.0)
                    .expect("synthesize")
            })
        };
        (
            cancelling.join().expect("cancelling thread"),
            keeping.join().expect("keeping thread"),
        )
    });

    let (seen, result) = cancelled;
    assert_eq!(seen, 2, "the sink was called past its failure");
    let err = result.expect_err("a failing sink must surface");
    assert!(err.to_string().contains("client went away"), "{err}");

    let alone = engine
        .synthesize(PROMPTS[0], ZERO_SHOT_VOICE_ID, 1.0)
        .expect("synthesize");
    assert_bit_identical(&kept, &alone, "row beside a cancelled one");
}
