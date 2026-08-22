//! Corpus-tokenization throughput for the NeuCodec encoder, CPU vs CUDA.
//!
//! Reports the metric that actually decides whether tokenizing a speech corpus
//! is practical: **realtime factor** (seconds of audio encoded per wall-clock
//! second). At 50 Hz codes, a 1000-hour corpus is 1000 / RTF hours of compute.
//!
//! Wall-clock is the right unit HERE, unlike for kernel-level A/B work, because
//! the question is throughput of a whole pipeline containing a host-side f64
//! frontend, H2D copies, and GPU kernels — instruction counts do not compose
//! across those. Treat the numbers as indicative and re-run on a quiet machine
//! before quoting them; the run prints a warning if the spread is wide.
//!
//! Usage:
//!   cargo run --release --example neucodec_tokenize_bench            # CPU only
//!   cargo run --release --features cuda --example neucodec_tokenize_bench
//!
//! Env:
//!   NEUCODEC_CHECKPOINT  path to model.safetensors (has a sensible default)
//!   NEUCODEC_BENCH_WAV   raw f32le 16 kHz mono samples to encode
//!   NEUCODEC_BENCH_SECS  seconds of audio per iteration (default 10)
//!   NEUCODEC_BENCH_ITERS timed iterations (default 5, plus 1 warmup)

use boostr::model::audio::neucodec::{NeuCodecEncoder, seamless_fbank};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;
use std::time::Instant;

const SAMPLE_RATE: usize = 16_000;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The benchmark input: a real waveform when one is supplied, else a
/// deterministic synthetic signal. Content does not change the work done — the
/// encoder is data-independent — but a real clip keeps the numbers honest if
/// anyone ever adds an early-exit.
fn samples(n: usize) -> Vec<f32> {
    if let Ok(path) = std::env::var("NEUCODEC_BENCH_WAV") {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let mut wave: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if wave.is_empty() {
            panic!("{path} contained no samples");
        }
        while wave.len() < n {
            let have = wave.len();
            wave.extend_from_within(..have.min(n - have));
        }
        wave.truncate(n);
        return wave;
    }
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1310.0 * t).sin()
        })
        .collect()
}

/// Resolve the checkpoint: `$NEUCODEC_CHECKPOINT`, else
/// `$BOOSTR_MODELS_DIR/neucodec/model.safetensors`. Returns a best-effort path
/// even when unresolved so `main` can report a precise error.
fn checkpoint() -> PathBuf {
    if let Ok(p) = std::env::var("NEUCODEC_CHECKPOINT") {
        return PathBuf::from(p);
    }
    match std::env::var("BOOSTR_MODELS_DIR") {
        Ok(root) => PathBuf::from(root).join("neucodec/model.safetensors"),
        Err(_) => PathBuf::new(),
    }
}

/// Run `iters` timed encodes (after one warmup) and report the realtime factor.
/// Prints min/median/max so a noisy machine is visible rather than averaged
/// into a confident-looking single number.
fn report(label: &str, secs: f64, mut times: Vec<f64>, frames: usize) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (min, max) = (times[0], times[times.len() - 1]);
    let median = times[times.len() / 2];
    let rtf = secs / median;
    println!(
        "{label:<6} median {median:>7.3}s  (min {min:.3} max {max:.3})  \
         RTF {rtf:>7.1}x  {frames} frames  ~{:.1} h/hour-of-compute",
        rtf,
    );
    if max > min * 1.5 {
        println!("       ^ spread >50%: machine is loaded, treat RTF as a lower bound");
    }
}

fn main() {
    let ckpt = checkpoint();
    if !ckpt.exists() {
        eprintln!(
            "checkpoint not found at {}; set NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR",
            ckpt.display()
        );
        std::process::exit(1);
    }
    let secs = env_usize("NEUCODEC_BENCH_SECS", 10);
    let iters = env_usize("NEUCODEC_BENCH_ITERS", 5);
    let wave = samples(secs * SAMPLE_RATE);
    println!("NeuCodec encode throughput — {secs}s of 16 kHz audio, {iters} timed iterations\n");

    {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let enc = NeuCodecEncoder::<CpuRuntime>::from_safetensors(&ckpt, &device)
            .expect("load encoder (cpu)");
        let warm = enc.encode(&client, &wave, &device).expect("cpu warmup");
        let frames = warm.shape()[2];
        let times = (0..iters)
            .map(|_| {
                let t = Instant::now();
                enc.encode(&client, &wave, &device).expect("cpu encode");
                t.elapsed().as_secs_f64()
            })
            .collect();
        report("cpu", secs as f64, times, frames);
    }

    #[cfg(feature = "cuda")]
    {
        use numr::runtime::Runtime;
        use numr::runtime::cuda::{CudaDevice, CudaRuntime};

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        let enc = NeuCodecEncoder::<CudaRuntime>::from_safetensors(&ckpt, &device)
            .expect("load encoder (cuda)");
        let warm = enc.encode(&client, &wave, &device).expect("cuda warmup");
        let frames = warm.shape()[2];
        let times = (0..iters)
            .map(|_| {
                let t = Instant::now();
                let out = enc.encode(&client, &wave, &device).expect("cuda encode");
                // Force completion before stopping the clock — CUDA launches are
                // async, so timing without a sync measures enqueue rate, not work.
                let _: Vec<i32> = out.contiguous().expect("contiguous").to_vec();
                t.elapsed().as_secs_f64()
            })
            .collect();
        report("cuda", secs as f64, times, frames);
    }

    // The mel frontend is host-side f64 and single-threaded by design, so it
    // costs the SAME on both backends. Measuring it separately says whether the
    // remaining time is GPU work worth optimizing or a serial frontend that
    // caps throughput no matter how fast the kernels get.
    {
        let device = CpuDevice::new();
        let times: Vec<f64> = (0..iters)
            .map(|_| {
                let t = Instant::now();
                seamless_fbank::<CpuRuntime>(&wave, &device).expect("fbank");
                t.elapsed().as_secs_f64()
            })
            .collect();
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        println!("\nof which — mel frontend (host f64, identical on both backends): {median:.3}s");
    }

    #[cfg(not(feature = "cuda"))]
    println!("\n(build with --features cuda to include the GPU path)");
}
