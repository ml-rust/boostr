//! Tests for [`super::measure_quality`].
//!
//! Test 6 cross-checks against `audio/corpus/manifests/sources.tsv`, the
//! real-corpus manifest referenced in the module docs. It is skipped (not
//! failed) when the corpus fixture or the manifest file is absent, matching
//! `crate::test_utils::corpus_flac`'s own skip-if-absent policy.

use super::*;
use crate::model::audio::decode_audio_file_mono_at;
use crate::test_utils::corpus_flac;
use std::path::{Path, PathBuf};

/// `sr / 10` matches [`super::measure_quality`]'s block length.
const RATE: u32 = 48_000;

fn sine(amplitude: f32, freq: f64, rate: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| amplitude * (std::f64::consts::TAU * freq * n as f64 / rate as f64).sin() as f32)
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
fn manifest_for(fixture: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AUDIO_CORPUS_MANIFEST") {
        let path = PathBuf::from(p);
        return path.exists().then_some(path);
    }
    let corpus_root = fixture.parent()?.parent()?.parent()?;
    let path = corpus_root.join("manifests").join("sources.tsv");
    path.exists().then_some(path)
}

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
