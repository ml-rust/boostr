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
mod tests {
    //! Tests for [`super::measure_quality`].
    //!
    //! Test 6 cross-checks against `audio/corpus/manifests/sources.tsv`, the
    //! real-corpus manifest referenced in the module docs. It is skipped (not
    //! failed) when the corpus fixture or the manifest file is absent, matching
    //! `crate::test_utils::corpus_flac`'s own skip-if-absent policy.

    use super::*;
    #[cfg(feature = "decode")]
    use crate::decode::decode_audio_file_mono_at;
    #[cfg(feature = "decode")]
    use crate::test_utils::corpus_flac;
    #[cfg(feature = "decode")]
    use std::path::{Path, PathBuf};

    /// `sr / 10` matches [`super::measure_quality`]'s block length.
    const RATE: u32 = 48_000;

    fn sine(amplitude: f32, freq: f64, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                amplitude * (std::f64::consts::TAU * freq * n as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    #[test]
    fn full_scale_sine_pins_peak_and_rms() {
        // 2 s so there are well over 20 blocks, keeping the floor computation
        // out of this test's assertions (test 4 below covers it directly).
        //
        // Amplitude is just under full scale on purpose. `clipped_samples` counts
        // `|s| >= 1.0`, which is the right detector for real PCM16 material — a
        // clipped sample sits exactly on the rail — but it also catches the peaks
        // of a mathematically exact full-scale sine, which are at the rail without
        // being clipped. Use 0.999 so this test pins level, not clipping.
        let samples = sine(0.999, 440.0, RATE, RATE as usize * 2);
        let q = measure_quality(&samples, RATE).expect("measure");

        assert!(
            (q.peak_dbfs - 0.0).abs() < 0.05,
            "peak_dbfs = {}",
            q.peak_dbfs
        );
        // 20*log10(1/sqrt(2)) = -3.0103...
        assert!(
            (q.rms_dbfs - (-3.0103)).abs() < 0.05,
            "rms_dbfs = {}",
            q.rms_dbfs
        );
        assert_eq!(q.clipped_samples, 0);
    }

    #[test]
    fn half_amplitude_sine_is_6_02_db_quieter() {
        let full = sine(1.0, 440.0, RATE, RATE as usize * 2);
        let half = sine(0.5, 440.0, RATE, RATE as usize * 2);

        let q_full = measure_quality(&full, RATE).expect("measure full");
        let q_half = measure_quality(&half, RATE).expect("measure half");

        // 20*log10(0.5) = -6.0206 dB, exactly, independent of the waveform shape.
        let delta = q_full.rms_dbfs - q_half.rms_dbfs;
        assert!((delta - 6.0206).abs() < 0.01, "delta = {delta}");
        let delta_peak = q_full.peak_dbfs - q_half.peak_dbfs;
        assert!(
            (delta_peak - 6.0206).abs() < 0.01,
            "delta_peak = {delta_peak}"
        );
    }

    #[test]
    fn clipping_count_is_exact() {
        // 100 samples at exactly +/-1.0 among 1000 quiet ones.
        let mut samples = vec![0.01f32; 900];
        for i in 0..100 {
            samples.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let q = measure_quality(&samples, RATE).expect("measure");
        assert_eq!(q.clipped_samples, 100);
    }

    #[test]
    fn noise_floor_tracks_the_quiet_blocks_not_the_loud_ones() {
        // 90 loud blocks then 10 near-silent blocks, each one block long, so the
        // sorted-block-list 5th percentile (index len/20) lands in the quiet run.
        // A percentile taken off the wrong end of the sorted list would report a
        // floor near the loud level instead, which this test must catch.
        let block_len = (RATE / 10) as usize;
        let mut samples = Vec::with_capacity(block_len * 100);
        for _ in 0..90 {
            samples.extend(sine(0.9, 440.0, RATE, block_len));
        }
        for _ in 0..10 {
            samples.extend(sine(0.001, 440.0, RATE, block_len));
        }
        // One extra sample so the loop's `n > block_len` block-count math has a
        // trailing partial block to drop, same as the reference.
        samples.push(0.0);

        let q = measure_quality(&samples, RATE).expect("measure");
        let loud_dbfs = dbfs(0.9 / std::f64::consts::SQRT_2);
        let quiet_dbfs = dbfs(0.001 / std::f64::consts::SQRT_2);

        assert!(
            q.floor_dbfs < (loud_dbfs + quiet_dbfs) / 2.0,
            "floor_dbfs {} should be near the quiet level {quiet_dbfs}, not the loud level {loud_dbfs}",
            q.floor_dbfs
        );
        assert!(q.snr_db > 0.0, "snr_db should be positive: {}", q.snr_db);
    }

    #[test]
    fn empty_and_zero_rate_return_err_not_panic() {
        assert!(measure_quality(&[], RATE).is_err());
        assert!(measure_quality(&[0.1, 0.2], 0).is_err());
    }

    #[test]
    fn all_silent_signal_is_finite_no_nan() {
        let samples = vec![0.0f32; RATE as usize];
        let q = measure_quality(&samples, RATE).expect("measure");

        assert!(q.peak_dbfs.is_finite());
        assert!(q.rms_dbfs.is_finite());
        assert!(q.floor_dbfs.is_finite());
        assert!(q.snr_db.is_finite());
        assert_eq!(q.peak_dbfs, SILENCE_DBFS);
        assert_eq!(q.rms_dbfs, SILENCE_DBFS);
        assert_eq!(q.floor_dbfs, SILENCE_DBFS);
        assert_eq!(q.snr_db, 0.0);
        assert_eq!(q.clipped_samples, 0);
    }

    /// One row of `audio/corpus/manifests/sources.tsv`.
    #[cfg(feature = "decode")]
    struct ManifestRow {
        file: String,
        duration_s: f64,
        floor_dbfs: f64,
        snr_db: f64,
    }

    /// Locate the manifest for a corpus fixture: `$AUDIO_CORPUS_MANIFEST`, else the
    /// sibling `manifests/sources.tsv` two levels above a `raw/<tier>/x.flac`
    /// fixture. Derived rather than hardcoded — an absolute path here would be a
    /// machine-specific reference committed into a published crate.
    #[cfg(feature = "decode")]
    fn manifest_for(fixture: &Path) -> Option<PathBuf> {
        if let Ok(p) = std::env::var("AUDIO_CORPUS_MANIFEST") {
            let path = PathBuf::from(p);
            return path.exists().then_some(path);
        }
        let corpus_root = fixture.parent()?.parent()?.parent()?;
        let path = corpus_root.join("manifests").join("sources.tsv");
        path.exists().then_some(path)
    }

    #[cfg(feature = "decode")]
    fn load_manifest_row(manifest_path: &Path, file_name: &str) -> Option<ManifestRow> {
        let text = std::fs::read_to_string(manifest_path).ok()?;
        for line in text.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 5 {
                continue;
            }
            if cols[0] == file_name {
                return Some(ManifestRow {
                    file: cols[0].to_string(),
                    duration_s: cols[2].parse().ok()?,
                    floor_dbfs: cols[3].parse().ok()?,
                    snr_db: cols[4].parse().ok()?,
                });
            }
        }
        None
    }

    /// Reads the fixture through the compressed-audio decoder, so it needs `decode`.
    #[cfg(feature = "decode")]
    #[test]
    fn matches_the_real_corpus_manifest_where_possible() {
        let Some(path) = corpus_flac() else { return };
        let Some(manifest_path) = manifest_for(&path) else {
            return;
        };
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            return;
        };
        let Some(row) = load_manifest_row(&manifest_path, file_name) else {
            return;
        };

        let samples = decode_audio_file_mono_at(&path, RATE).expect("decode corpus fixture");
        let q = measure_quality(&samples, RATE).expect("measure corpus fixture");

        assert!(
            (q.duration_s - row.duration_s).abs() < 0.5,
            "duration_s {} not within 0.5s of manifest's {} for {}",
            q.duration_s,
            row.duration_s,
            row.file
        );

        // floor_dbfs/snr_db in the manifest were NOT confirmed to come from this
        // module's reference (record.py): record.py only decodes WAV via
        // Python's `wave` module and never touches the FLAC sources this
        // manifest lists, and no other tool producing these columns was found
        // in the audio/ repo (see module docs). Assert only sane-range
        // properties here; a numeric tolerance against the manifest would need
        // the original tool's definitions confirmed first.
        assert!(
            q.floor_dbfs < q.rms_dbfs,
            "floor_dbfs {} should be below rms for a real recording",
            q.floor_dbfs
        );
        assert!(q.snr_db > 0.0, "snr_db {} should be positive", q.snr_db);
        let _ = row.floor_dbfs;
        let _ = row.snr_db;
    }
}
