//! Shared fixtures: a sample rate, a noise generator and the phrase/gap
//! timing the enhancement-chain tests build synthetic takes from, plus the
//! model-fixture resolution the checkpoint-gated tests skip on.

// Each file in `tests/` compiles as its own crate and pulls in this module
// wholesale, so any helper a given test crate does not call reads as dead
// code there. The helpers ARE used — just not all of them by every crate.
#![allow(dead_code)]

pub mod signals;

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
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            let root = std::env::var("BOOSTR_MODELS_DIR")
                .ok()
                .filter(|r| !r.is_empty())?;
            let root = PathBuf::from(root);
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

/// Shape + summary statistics for a tensor, computed in f64. Comparable
/// across a Python oracle script and this crate regardless of which side
/// produced it. Used by `whisper_word_oracle` to diagnose mel/encoder drift
/// without asserting on it (that crate's product surface is the alignment
/// matrix, not the mel front end).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl TensorStats {
    pub fn from_data(shape: &[usize], data: &[f32]) -> Self {
        let n = data.len().max(1) as f64;
        let sum: f64 = data.iter().map(|&x| x as f64).sum();
        let mean = sum / n;
        let var: f64 = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        let min = data.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        Self {
            shape: shape.to_vec(),
            mean,
            std: var.sqrt(),
            min,
            max,
        }
    }
}

/// Print one tensor's stats next to a reference's, if the reference carries
/// one. Diagnostic only — never asserts, since the caller's own product
/// surface is elsewhere.
pub fn report_tensor_stats(
    label: &str,
    ours: &TensorStats,
    theirs: Option<&TensorStats>,
    unavailable_hint: &str,
) {
    let Some(theirs) = theirs else {
        eprintln!("{label} stats: ours = {ours:?}; reference carries none ({unavailable_hint})");
        return;
    };
    eprintln!(
        "{label} stats: ours = {ours:?}; reference = {theirs:?}; shapes match = {}, mean diff = \
         {:.6}, std diff = {:.6}, min diff = {:.6}, max diff = {:.6}",
        ours.shape == theirs.shape,
        (ours.mean - theirs.mean).abs(),
        (ours.std - theirs.std).abs(),
        (ours.min - theirs.min).abs(),
        (ours.max - theirs.max).abs(),
    );
}

/// Per-row/per-frame diagnostic for two equal-length flat `[rows, frames]`
/// planes: where the biggest divergence is and how it compares to the row's
/// own scale, plus a per-row summary so a reviewer can tell "one bad row"
/// from "drift spread evenly across all of them". Shared by
/// `whisper_word_oracle`'s per-head raw-plane and alignment-matrix checks.
pub struct PlaneDiag {
    pub max_diff: f32,
    pub max_row: usize,
    pub max_frame: usize,
    pub mean_diff: f32,
    pub oracle_at_max: f32,
    pub ours_at_max: f32,
    pub row_max_value: f32,
    pub per_row_max_diff: Vec<f32>,
}

pub fn diff_plane(ours: &[f32], oracle: &[f32], frames: usize) -> PlaneDiag {
    let rows = ours.len() / frames.max(1);
    let mut per_row_max_diff = vec![0.0f32; rows];
    let mut max_diff = 0.0f32;
    let mut max_row = 0;
    let mut max_frame = 0;
    let mut sum_diff = 0.0f64;

    for (r, row_max) in per_row_max_diff.iter_mut().enumerate() {
        for f in 0..frames {
            let i = r * frames + f;
            let d = (ours[i] - oracle[i]).abs();
            sum_diff += d as f64;
            if d > *row_max {
                *row_max = d;
            }
            if d > max_diff {
                max_diff = d;
                max_row = r;
                max_frame = f;
            }
        }
    }

    let row_max_value = ours[max_row * frames..(max_row + 1) * frames]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    PlaneDiag {
        max_diff,
        max_row,
        max_frame,
        mean_diff: (sum_diff / ours.len().max(1) as f64) as f32,
        oracle_at_max: oracle[max_row * frames + max_frame],
        ours_at_max: ours[max_row * frames + max_frame],
        row_max_value,
        per_row_max_diff,
    }
}

/// Pushes one failure onto `failures` per index pair whose value differs,
/// naming `what`. Shared by any oracle test comparing an exact integer path
/// (e.g. a DTW backtrace) against a fixture.
pub fn check_exact_usize(
    label: &str,
    what: &str,
    got: &[usize],
    want: &[usize],
    failures: &mut Vec<String>,
) {
    if got != want {
        failures.push(format!(
            "{label}: {what} differs from the oracle ({} entries vs {})",
            got.len(),
            want.len()
        ));
    }
}

/// Pushes a failure onto `failures` when `got` and `want` differ in length or
/// their max absolute difference reaches `tol`. Shared by any oracle test
/// comparing a float array (e.g. per-token timestamps) against a fixture.
pub fn check_close_f32(
    label: &str,
    what: &str,
    got: &[f32],
    want: &[f32],
    tol: f32,
    failures: &mut Vec<String>,
) {
    if got.len() != want.len() {
        failures.push(format!(
            "{label}: {what} length ({}) differs from the oracle ({})",
            got.len(),
            want.len()
        ));
        return;
    }
    let max_diff = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    eprintln!("{label}: {what} max abs diff = {max_diff}");
    if max_diff >= tol {
        failures.push(format!(
            "{label}: {what} differs from the oracle by {max_diff}, over {tol}"
        ));
    }
}

/// Pushes one failure onto `failures` per word whose text or (start, end)
/// timing differs from the oracle's, beyond `time_tol`. `got`/`want` are
/// `(text, start, end)` triples — a whisper-word-agnostic shape so this
/// helper carries no dependency on any particular word-timing type.
pub fn check_word_timings(
    label: &str,
    got: &[(String, f32, f32)],
    want: &[(String, f32, f32)],
    time_tol: f32,
    failures: &mut Vec<String>,
) {
    if got.len() != want.len() {
        failures.push(format!(
            "{label}: word count ({}) differs from the oracle ({})",
            got.len(),
            want.len()
        ));
        return;
    }
    for (i, ((text, start, end), (want_text, want_start, want_end))) in
        got.iter().zip(want.iter()).enumerate()
    {
        if text != want_text {
            failures.push(format!(
                "{label}: word {i} text {text:?} differs from the oracle {want_text:?}"
            ));
        }
        if (start - want_start).abs() >= time_tol || (end - want_end).abs() >= time_tol {
            failures.push(format!(
                "{label}: word {i} ({text:?}) timing ({start}, {end}) differs from the oracle \
                 ({want_start}, {want_end}), over {time_tol}"
            ));
        }
    }
}
