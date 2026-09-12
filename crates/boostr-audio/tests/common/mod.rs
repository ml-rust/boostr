//! Shared fixtures: a sample rate, a noise generator and the phrase/gap
//! timing the enhancement-chain tests build synthetic takes from, plus the
//! model-fixture resolution the checkpoint-gated tests skip on.

// Each file in `tests/` compiles as its own crate and pulls in this module
// wholesale, so any helper a given test crate does not call reads as dead
// code there. The helpers ARE used — just not all of them by every crate.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use numr::runtime::cpu::{CpuClient, CpuDevice};
use serde::de::DeserializeOwned;

pub const RATE: u32 = 48_000;

/// A deterministic pseudo-random hiss in `[-amp, amp]`.
///
/// A fixed LCG rather than a crate: the test must fail or pass identically on
/// every machine, and a seeded generator is the only way to assert a specific
/// dB improvement.
pub fn hiss(n: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            (u as f32) * amp
        })
        .collect()
}

/// Phrase length and gap length, in samples. Real speech is bursts separated
/// by pauses, and the pauses are what a gate measures its floor from.
pub const PHRASE: usize = RATE as usize * 4 / 5;
pub const GAP: usize = RATE as usize * 2 / 5;

/// True where a sample falls inside a spoken phrase.
pub fn is_voiced(i: usize) -> bool {
    i % (PHRASE + GAP) < PHRASE
}

pub fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Resolve a model fixture: `$<specific_var>`, else `$BOOSTR_MODELS_DIR/<relative>`
/// (pass `relative = ""` when the specific var's fallback is the models root
/// itself). Returns `None` when neither is set or the resolved path is absent
/// on disk, so callers skip.
pub fn model_fixture(specific_var: &str, relative: &str) -> Option<PathBuf> {
    let path = match std::env::var(specific_var) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let root = PathBuf::from(std::env::var("BOOSTR_MODELS_DIR").ok()?);
            if relative.is_empty() {
                root
            } else {
                root.join(relative)
            }
        }
    };
    path.exists().then_some(path)
}

/// Print the standard skip notice naming both the specific env var and
/// `BOOSTR_MODELS_DIR`, so the reader knows how to enable the test.
pub fn skip_notice(what: &str, specific_var: &str) {
    eprintln!("skipping: {what} unavailable; set {specific_var} or BOOSTR_MODELS_DIR");
}

/// Read and parse `path` as JSON, panicking with the path on either failure.
/// Shared by the reference-file loaders in `whisper_integration` and
/// `silero_vad_segment_parity` — same two steps, different `Reference` types.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> T {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}
