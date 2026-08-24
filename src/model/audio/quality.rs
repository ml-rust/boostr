//! Take-quality metrics for recorded/decoded speech: peak, RMS, noise floor, SNR.
//!
//! Rust port of `audio/pipeline/record.py`'s `dbfs()` and `measure()` (the
//! Python voice-corpus recorder). Definitions matched to that script because
//! `audio/corpus/manifests/sources.tsv` — the ground-truth manifest checked
//! against in tests — was committed in the same commit as `record.py` and no
//! other tool in that repository computes `floor_dbfs`/`snr_db`; `record.py`
//! itself only reads WAV via Python's `wave` module and never touches the
//! FLAC sources the manifest lists, so the manifest's exact generator is not
//! present in the repo. `record.py` is therefore the best-available reference,
//! not a confirmed match for the manifest.
//!
//! **CPU-only.** Two sequential passes over `samples` (one for the running
//! sum of squares and max, one for the per-block RMS histogram) — there is
//! no batching or matrix structure for a GPU kernel to exploit, so this
//! stays plain scalar loops like `super::resample`.

use crate::error::{Error, Result};

/// dBFS floor used in place of `-inf` for a zero (or non-positive) amplitude.
///
/// `record.py` returns `float("-inf")` for `dbfs(0)`. This module cannot: an
/// all-silent signal has both `rms == 0` and `floor == 0`, and
/// `snr_db = rms_dbfs - floor_dbfs` would then be `-inf - -inf = NaN`. A
/// large-but-finite sentinel, far below any real recording's noise floor,
/// keeps every field finite while still reading as "silence" to a caller.
const SILENCE_DBFS: f64 = -600.0;

/// `20 * log10(x)`, floored at [`SILENCE_DBFS`] for `x <= 0` instead of `-inf`.
fn dbfs(x: f64) -> f64 {
    if x > 0.0 {
        20.0 * x.log10()
    } else {
        SILENCE_DBFS
    }
}

/// Measured acoustics of one recording, in the same terms the corpus manifest uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TakeQuality {
    /// Length of `samples` in seconds, i.e. `samples.len() / sample_rate`.
    pub duration_s: f64,
    /// `20 * log10(max |s|)`.
    pub peak_dbfs: f64,
    /// `20 * log10(rms(s))` over the whole signal.
    pub rms_dbfs: f64,
    /// `20 * log10` of the 5th-percentile block RMS (see module docs).
    pub floor_dbfs: f64,
    /// `rms_dbfs - floor_dbfs`.
    pub snr_db: f64,
    /// Count of samples with `|s| >= 1.0`.
    pub clipped_samples: usize,
}

/// Measure `samples` at `sample_rate`. Mono, `[-1, 1]`.
///
/// The noise floor is the 5th percentile of per-block RMS, blocks being
/// non-overlapping 100 ms windows (`sample_rate / 10` samples), matching
/// `record.py`'s `blk = sr // 10` and `floor = blocks[len(blocks) // 20]`
/// after sorting ascending. As in the reference, a trailing partial block is
/// dropped rather than measured. When `samples` is shorter than one block
/// there are no blocks at all; `floor_dbfs` then falls back to
/// [`SILENCE_DBFS`] (documented above) rather than the reference's `-inf`.
/// When there is at least one block but fewer than 20, the floor is the
/// single quietest block (`blocks[0]`), also matching the reference.
///
/// Returns [`Error::InvalidArgument`] when `samples` is empty or
/// `sample_rate` is 0. Never panics and never produces `NaN`: an all-silent
/// input yields `peak_dbfs == rms_dbfs == floor_dbfs == `[`SILENCE_DBFS`]`
/// and `snr_db == 0.0`.
pub fn measure_quality(samples: &[f32], sample_rate: u32) -> Result<TakeQuality> {
    if samples.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: "sample slice is empty".to_string(),
        });
    }
    if sample_rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "sample_rate",
            reason: "sample rate is 0".to_string(),
        });
    }

    let mut peak = 0.0f64;
    let mut sum_sq = 0.0f64; // f64 accumulator: a 26-minute file is ~76M samples.
    let mut clipped = 0usize;
    for &s in samples {
        let a = (s as f64).abs();
        if a > peak {
            peak = a;
        }
        sum_sq += a * a;
        if a >= 1.0 {
            clipped += 1;
        }
    }
    let n = samples.len();
    let rms = (sum_sq / n as f64).sqrt();

    // 100 ms blocks (sample_rate / 10 samples), at least 1 sample so a
    // pathologically low sample_rate can't produce a zero-length step.
    let block_len = ((sample_rate / 10) as usize).max(1);
    let mut blocks: Vec<f64> = Vec::new();
    if n > block_len {
        let stop = n - block_len;
        let mut i = 0usize;
        while i < stop {
            let block = &samples[i..i + block_len];
            let block_sum_sq: f64 = block.iter().map(|&s| (s as f64) * (s as f64)).sum();
            blocks.push((block_sum_sq / block_len as f64).sqrt());
            i += block_len;
        }
    }
    blocks.sort_by(f64::total_cmp);
    let floor = if blocks.len() >= 20 {
        blocks[blocks.len() / 20]
    } else if let Some(&min) = blocks.first() {
        min
    } else {
        0.0
    };

    let peak_dbfs = dbfs(peak);
    let rms_dbfs = dbfs(rms);
    let floor_dbfs = dbfs(floor);

    Ok(TakeQuality {
        duration_s: n as f64 / sample_rate as f64,
        peak_dbfs,
        rms_dbfs,
        floor_dbfs,
        snr_db: rms_dbfs - floor_dbfs,
        clipped_samples: clipped,
    })
}

#[cfg(test)]
mod tests;
