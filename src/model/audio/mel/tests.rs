//! Tests for the mel front end.
//!
//! Numerical parity against HuggingFace's `WhisperFeatureExtractor` lives in
//! `tests/whisper_mel_parity.rs`, which needs an out-of-repo fixture. These
//! tests pin the parts that can be checked in isolation: the two mel scales,
//! the filterbank shape, and the framing arithmetic.

use super::*;

#[test]
fn test_hz_mel_roundtrip() {
    let hz = 1000.0;
    let mel = hz_to_mel(hz);
    let recovered = mel_to_hz(mel);
    assert!(
        (recovered - hz).abs() < 0.01,
        "roundtrip failed: {recovered}"
    );
}

#[test]
fn test_mel_frequencies_count() {
    let freqs = mel_frequencies(80, 0.0, 8000.0);
    assert_eq!(freqs.len(), 82); // num_mel_bins + 2
    assert!((freqs[0] - 0.0).abs() < 1.0);
}

#[test]
fn test_spectrogram_shape() {
    // 1 second of silence at 16kHz
    let samples = vec![0.0f32; 16000];
    let result = compute_mel_spectrogram(&samples, 128, 16000).expect("mel");
    let num_frames = (16000 - 400) / 160 + 1; // 98
    assert_eq!(result.len(), 128 * num_frames);
}

#[test]
fn test_spectrogram_short_audio() {
    // Too short for even one frame
    let samples = vec![0.0f32; 100];
    let result = compute_mel_spectrogram(&samples, 80, 16000).expect("mel");
    assert!(result.is_empty());
}

// --- Slaney mel scale -------------------------------------------------------

#[test]
fn slaney_linear_region_is_hz_over_f_sp() {
    // Below 1 kHz the scale is exactly linear: 200/3 Hz per mel.
    assert!((hz_to_mel_slaney(200.0) - 3.0).abs() < 1e-12);
    assert!((hz_to_mel_slaney(0.0)).abs() < 1e-12);
    assert!((mel_to_hz_slaney(3.0) - 200.0).abs() < 1e-9);
}

#[test]
fn slaney_breakpoint_is_15_mels() {
    // 1000 Hz sits exactly on the linear/log join, at mel 15.
    assert!((hz_to_mel_slaney(1000.0) - 15.0).abs() < 1e-12);
    assert!((mel_to_hz_slaney(15.0) - 1000.0).abs() < 1e-9);
}

#[test]
fn slaney_log_region_spans_6p4x_over_27_mels() {
    // By construction 6.4 kHz is 27 mels above the 1 kHz breakpoint.
    assert!((hz_to_mel_slaney(6400.0) - 42.0).abs() < 1e-9);
    assert!((mel_to_hz_slaney(42.0) - 6400.0).abs() < 1e-6);
}

#[test]
fn slaney_roundtrips_across_both_regions() {
    for hz in [0.0, 50.0, 500.0, 999.0, 1000.0, 2000.0, 8000.0] {
        let back = mel_to_hz_slaney(hz_to_mel_slaney(hz));
        assert!((back - hz).abs() < 1e-6, "roundtrip {hz} -> {back}");
    }
}

#[test]
fn slaney_and_htk_edges_differ() {
    let htk = mel_frequencies_with(80, 0.0, 8000.0, MelScale::Htk);
    let slaney = mel_frequencies_with(80, 0.0, 8000.0, MelScale::Slaney);
    assert_eq!(htk.len(), 82);
    assert_eq!(slaney.len(), 82);
    // Endpoints coincide; the interior warping does not.
    assert!((htk[0] - slaney[0]).abs() < 1e-9);
    assert!((htk[81] - slaney[81]).abs() < 1e-6);
    assert!(
        (htk[40] - slaney[40]).abs() > 1.0,
        "htk {} vs slaney {}",
        htk[40],
        slaney[40]
    );
}

// --- Filterbank -------------------------------------------------------------

#[test]
fn slaney_norm_scales_filters_by_inverse_span() {
    let plain = mel_filterbank(
        80,
        400,
        16000.0,
        0.0,
        8000.0,
        MelScale::Slaney,
        MelNorm::None,
    );
    let normed = mel_filterbank(
        80,
        400,
        16000.0,
        0.0,
        8000.0,
        MelScale::Slaney,
        MelNorm::Slaney,
    );
    let edges = mel_frequencies_with(80, 0.0, 8000.0, MelScale::Slaney);
    let bins = 400 / 2 + 1;
    assert_eq!(plain.len(), 80 * bins);
    for m in 0..80 {
        let enorm = 2.0 / (edges[m + 2] - edges[m]);
        for k in 0..bins {
            let a = plain[m * bins + k] * enorm;
            let b = normed[m * bins + k];
            assert!((a - b).abs() < 1e-12, "filter {m} bin {k}: {a} vs {b}");
        }
    }
}

#[test]
fn filterbank_weights_are_non_negative_and_bounded() {
    let fb = mel_filterbank(80, 400, 16000.0, 0.0, 8000.0, MelScale::Htk, MelNorm::None);
    for w in &fb {
        assert!((0.0..=1.0).contains(w), "weight out of range: {w}");
    }
}

// --- Framing ----------------------------------------------------------------

#[test]
fn whisper_options_always_produce_3000_frames() {
    let opts = MelOptions::whisper(80, 16000);
    // Half a second of silence: pad-to-30s then centered framing.
    let half_second = vec![0.0f32; 8000];
    let short = compute_mel_spectrogram_with(&half_second, 16000, &opts).expect("mel");
    assert_eq!(short.len(), 80 * 3000);
    // Longer than 30 s: truncated to the same length.
    let over_thirty = vec![0.0f32; 500_000];
    let long = compute_mel_spectrogram_with(&over_thirty, 16000, &opts).expect("mel");
    assert_eq!(long.len(), 80 * 3000);
}

#[test]
fn whisper_log_floor_is_eight_below_the_global_max() {
    // A tone in the first half, silence in the second. The silent frames must
    // sit at the global floor, which is (max - 8 + 4) / 4 in output units —
    // a per-frame max would instead lift them back to the tone's level.
    let mut samples = vec![0.0f32; 16000 * 30];
    for (i, s) in samples.iter_mut().take(16000).enumerate() {
        *s = (2.0 * PI * 440.0 * i as f32 / 16000.0).sin();
    }
    let opts = MelOptions::whisper(80, 16000);
    let mel = compute_mel_spectrogram_with(&samples, 16000, &opts).expect("mel");
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = mel.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        (max - min - 2.0).abs() < 1e-4,
        "span should be exactly 8 log10 units / 4 = 2.0, got {}",
        max - min
    );
}

#[test]
fn rejects_win_length_longer_than_n_fft() {
    let mut opts = MelOptions::new(80);
    opts.win_length = 512;
    assert!(compute_mel_spectrogram_with(&[0.0f32; 4000], 16000, &opts).is_err());
}

#[test]
fn rejects_zero_sized_parameters() {
    let mut opts = MelOptions::new(80);
    opts.hop_length = 0;
    assert!(compute_mel_spectrogram_with(&[0.0f32; 4000], 16000, &opts).is_err());

    let zero_bins = MelOptions::new(0);
    assert!(compute_mel_spectrogram_with(&[0.0f32; 4000], 16000, &zero_bins).is_err());
}

// --- power_spectra ----------------------------------------------------------

// Ported from the deleted `fft.rs` (which hand-rolled a radix-2 FFT):
// known-signal power spectrum checks against `power_spectra`, numr's
// batched `rfft` now doing the work. Dropped `rejects_non_power_of_two`
// — it asserted the radix-2 implementation's own panic message, which no
// longer exists now that numr's fallible `rfft` is behind it.

fn naive_dft_power(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let bins = n / 2 + 1;
    let mut out = Vec::with_capacity(bins);
    for k in 0..bins {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, &v) in x.iter().enumerate() {
            let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
            re += v * angle.cos();
            im += v * angle.sin();
        }
        out.push(re * re + im * im);
    }
    out
}

#[test]
fn power_spectra_dc_signal() {
    let x = vec![1.0f32; 8];
    let p = power_spectra(&x, 1, 8).expect("rfft");
    // All energy in bin 0: |sum|^2 = 64
    assert!((p[0] - 64.0).abs() < 1e-3);
    for v in &p[1..] {
        assert!(v.abs() < 1e-3);
    }
}

#[test]
fn power_spectra_matches_naive_dft_small() {
    // Arbitrary waveform, size 16 (power of 2).
    let x: Vec<f32> = (0..16)
        .map(|i| (0.3 * i as f32).sin() + 0.5 * (0.7 * i as f32).cos())
        .collect();
    let fft = power_spectra(&x, 1, 16).expect("rfft");
    let dft = naive_dft_power(&x);
    assert_eq!(fft.len(), dft.len());
    for (a, b) in fft.iter().zip(dft.iter()) {
        assert!((a - b).abs() < 1e-3, "fft {a} != dft {b}");
    }
}

#[test]
fn power_spectra_matches_naive_dft_512() {
    let x: Vec<f32> = (0..512)
        .map(|i| (0.01 * i as f32).sin() + 0.3 * ((0.05 * i as f32).cos()))
        .collect();
    let fft = power_spectra(&x, 1, 512).expect("rfft");
    let dft = naive_dft_power(&x);
    assert_eq!(fft.len(), 257);
    // Relative tolerance per-bin since absolute magnitudes vary.
    // Widened from the radix-2 test's 1e-3 to 1e-2: numr's rfft
    // accumulates 512-point sums in a different order (and via F32
    // Complex64 intermediates) than this naive O(n^2) DFT reference,
    // so per-bin drift is legitimately larger while still tiny.
    for (a, b) in fft.iter().zip(dft.iter()) {
        let denom = b.abs().max(1.0);
        assert!((a - b).abs() / denom < 1e-2, "fft {a} vs dft {b}");
    }
}

#[test]
fn power_spectra_matches_naive_dft_at_n_fft_400() {
    // Non-power-of-two: numr routes this through Bluestein. Whisper's exact
    // FFT size, so this pins the path the parity test depends on.
    let x: Vec<f32> = (0..400)
        .map(|i| (0.02 * i as f32).sin() + 0.4 * ((0.11 * i as f32).cos()))
        .collect();
    let fft = power_spectra(&x, 1, 400).expect("rfft");
    let dft = naive_dft_power(&x);
    assert_eq!(fft.len(), 201);
    for (a, b) in fft.iter().zip(dft.iter()) {
        let denom = b.abs().max(1.0);
        assert!((a - b).abs() / denom < 1e-2, "fft {a} vs dft {b}");
    }
}
