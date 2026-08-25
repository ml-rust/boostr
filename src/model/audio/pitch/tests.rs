//! Tests for [`super::estimate_pitch`].
//!
//! Every frequency assertion is pinned against the generator's own f0, so a
//! regression that reports a harmonic or a sub-harmonic fails loudly rather
//! than merely producing a finite number.

use super::*;

const RATE: u32 = 16_000;

/// Sine of `freq` Hz, `secs` long, amplitude 0.8.
fn sine(freq: f64, secs: f64) -> Vec<f32> {
    let n = (secs * RATE as f64) as usize;
    (0..n)
        .map(|i| 0.8 * (std::f64::consts::TAU * freq * i as f64 / RATE as f64).sin() as f32)
        .collect()
}

/// Six harmonics of `f0` with `1/k` amplitudes — a sawtooth-like glottal
/// shape. This is the signal that separates YIN from autocorrelation.
fn harmonic_stack(f0: f64, secs: f64) -> Vec<f32> {
    let n = (secs * RATE as f64) as usize;
    let norm: f64 = (1..=6).map(|k| 1.0 / k as f64).sum();
    (0..n)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            let v: f64 = (1..=6)
                .map(|k| (std::f64::consts::TAU * f0 * k as f64 * t).sin() / k as f64)
                .sum();
            (0.9 * v / norm) as f32
        })
        .collect()
}

/// Deterministic white noise in `[-0.5, 0.5]` from a 64-bit LCG.
fn white_noise(secs: f64) -> Vec<f32> {
    let n = (secs * RATE as f64) as usize;
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f64 / (1u64 << 24) as f64 - 0.5) as f32
        })
        .collect()
}

fn track(samples: &[f32]) -> PitchTrack {
    estimate_pitch(samples, RATE, PitchOptions::default()).expect("estimate_pitch")
}

fn mean(samples: &[f32]) -> f64 {
    track(samples).mean_hz.expect("voiced frames")
}

#[test]
fn pure_tones_across_the_range_are_within_one_percent() {
    for expected in [100.0, 220.0, 300.0] {
        let t = track(&sine(expected, 1.0));
        let got = t.mean_hz.expect("voiced frames");
        assert!(
            (got - expected).abs() / expected < 0.01,
            "{expected} Hz sine reported {got} Hz"
        );
        assert!(
            t.voiced_fraction > 0.9,
            "{expected} Hz sine voiced_fraction = {}",
            t.voiced_fraction
        );
    }
}

#[test]
fn harmonic_rich_signal_reports_f0_not_a_harmonic() {
    // A harmonic stack is where a naive estimator reports 240 or 360 Hz rather
    // than the true 120 Hz f0, so this pins the octave decision on a
    // speech-shaped signal instead of a bare sine.
    //
    // It does NOT isolate YIN's cumulative-mean-normalisation step — verified
    // by sabotage: replacing `d * tau / running` with a plain rescale of `d`
    // leaves this test passing, because rescaling does not move the minima of
    // the difference function. `silence_and_noise_are_unvoiced` is the test
    // that fails under that sabotage, since the 0.15 threshold is only
    // meaningful against a scale-invariant `d'`.
    let got = mean(&harmonic_stack(120.0, 1.0));
    assert!(
        (got - 120.0).abs() / 120.0 < 0.02,
        "harmonic stack at 120 Hz reported {got} Hz"
    );
    assert!(
        (got - 240.0).abs() / 240.0 > 0.10,
        "harmonic stack at 120 Hz reported the second harmonic: {got} Hz"
    );
}

#[test]
fn high_tone_does_not_drop_an_octave() {
    let got = mean(&sine(300.0, 1.0));
    assert!(
        (got - 150.0).abs() / 150.0 > 0.10,
        "300 Hz sine reported the sub-harmonic: {got} Hz"
    );
    assert!(
        (got - 300.0).abs() / 300.0 < 0.01,
        "300 Hz sine got {got} Hz"
    );
}

/// This is the test that pins YIN's cumulative mean normalised difference.
/// `d'` is scale-invariant, which is what makes a fixed 0.15 threshold mean
/// "aperiodic" regardless of level. Sabotage-verified: rescale `d` by any
/// amplitude-dependent factor instead and noise starts reporting voiced frames.
#[test]
fn silence_and_noise_are_unvoiced() {
    let silence = track(&vec![0.0f32; RATE as usize]);
    assert_eq!(silence.mean_hz, None);
    assert_eq!(silence.std_hz, None);
    assert_eq!(silence.voiced_fraction, 0.0);
    assert!(silence.f0.iter().all(|f| f.is_none()));

    let noise = track(&white_noise(1.0));
    assert!(
        noise.voiced_fraction < 0.1,
        "white noise voiced_fraction = {}",
        noise.voiced_fraction
    );
}

#[test]
fn non_integer_period_needs_parabolic_interpolation() {
    // Chosen to sit exactly HALFWAY between two integer lags, the worst case
    // for a lag-quantised estimator: 16000 / 316.8317 = 50.5 samples. Whichever
    // way an uninterpolated search rounds, it is ~1% out (lag 50 -> 320.0 Hz,
    // lag 51 -> 313.7 Hz), so the 0.5% bound below cannot be met without
    // parabolic interpolation.
    //
    // Do not soften this to 1%: at 1% a rounded lag still passes, and the test
    // stops testing the thing it is named after.
    const F0: f64 = 16_000.0 / 50.5;
    let got = mean(&sine(F0, 1.0));
    assert!(
        (got - F0).abs() / F0 < 0.005,
        "{F0} Hz sine reported {got} Hz — a lag-quantised estimate would be ~1% out"
    );
}

#[test]
fn std_hz_tracks_real_pitch_variation() {
    let flat = track(&sine(200.0, 1.0));
    let flat_std = flat.std_hz.expect("two voiced frames");
    assert!(flat_std < 2.0, "constant 200 Hz tone std_hz = {flat_std}");

    // 1 s at 150 Hz then 1 s at 250 Hz, phase-continuous so the join adds no
    // click. Mean of the two halves is 200; population std is 50.
    let n = RATE as usize;
    let mut phase = 0.0f64;
    let stepped: Vec<f32> = (0..2 * n)
        .map(|i| {
            let f = if i < n { 150.0 } else { 250.0 };
            phase += std::f64::consts::TAU * f / RATE as f64;
            0.8 * phase.sin() as f32
        })
        .collect();
    let t = track(&stepped);
    let m = t.mean_hz.expect("voiced frames");
    let s = t.std_hz.expect("two voiced frames");
    assert!((m - 200.0).abs() < 10.0, "stepped mean_hz = {m}");
    assert!((40.0..60.0).contains(&s), "stepped std_hz = {s}");
}

#[test]
fn invalid_arguments_are_rejected() {
    let samples = sine(200.0, 0.5);
    let d = PitchOptions::default();
    assert!(estimate_pitch(&samples, 0, d).is_err(), "sample_rate == 0");
    assert!(estimate_pitch(&[], RATE, d).is_err(), "empty samples");

    let cases = [
        PitchOptions {
            min_hz: 400.0,
            max_hz: 60.0,
            ..d
        },
        PitchOptions { min_hz: 0.0, ..d },
        PitchOptions { window_s: 0.0, ..d },
        PitchOptions { hop_s: 0.0, ..d },
        // 0.020 s spans less than two periods of 60 Hz (0.0333 s).
        PitchOptions {
            window_s: 0.020,
            ..d
        },
        // Nyquist at 16 kHz is 8 kHz.
        PitchOptions {
            max_hz: 9_000.0,
            ..d
        },
    ];
    for opts in cases {
        assert!(
            estimate_pitch(&samples, RATE, opts).is_err(),
            "expected rejection for {opts:?}"
        );
    }
}

#[test]
fn input_shorter_than_one_window_yields_an_empty_track() {
    // 0.045 s window at 16 kHz is 720 samples; 100 is well short.
    let t = estimate_pitch(&sine(200.0, 0.0), RATE, PitchOptions::default());
    assert!(
        t.is_err(),
        "zero-length input is still an empty-input error"
    );

    let t = track(&sine(200.0, 0.004));
    assert!(t.f0.is_empty(), "f0 = {:?}", t.f0);
    assert_eq!(t.mean_hz, None);
    assert_eq!(t.std_hz, None);
    assert_eq!(t.voiced_fraction, 0.0);
}

// --- aperiodicity / HNR -----------------------------------------------------
//
// These pin the continuous periodicity measure YIN already computes. Before it
// was exposed, `voiced_fraction` was the only signal available and it collapses
// a clean tone and a barely-voiced hiss into the same "voiced" bucket.

#[test]
fn a_pure_tone_is_far_more_harmonic_than_the_same_tone_with_noise() {
    let rate = 16_000usize;
    let f0 = 150.0f64;
    let n = rate; // one second

    let clean: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 / rate as f64).sin() as f32)
        .collect();

    // Deterministic additive noise at ~30% amplitude. An LCG, not a constant,
    // so it is broadband rather than a second tone.
    let mut state = 0x1234_5678u32;
    let noisy: Vec<f32> = clean
        .iter()
        .map(|&s| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let r = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
            s + r * 0.6
        })
        .collect();

    let clean_track = estimate_pitch(&clean, rate as u32, PitchOptions::default()).unwrap();
    let noisy_track = estimate_pitch(&noisy, rate as u32, PitchOptions::default()).unwrap();

    let clean_hnr = clean_track.mean_hnr_db.expect("clean tone must be voiced");
    let noisy_hnr = noisy_track.mean_hnr_db.expect("noisy tone must be voiced");

    assert!(
        clean_hnr > noisy_hnr + 5.0,
        "clean HNR {clean_hnr:.1} dB should exceed noisy {noisy_hnr:.1} dB by >5 dB"
    );

    // Both are still detected as voiced at the same F0, which is the whole
    // point: voicing alone cannot separate them, HNR can.
    let clean_f0 = clean_track.mean_hz.unwrap();
    let noisy_f0 = noisy_track.mean_hz.unwrap();
    assert!((clean_f0 - f0).abs() < 5.0, "clean f0 {clean_f0}");
    assert!((noisy_f0 - f0).abs() < 5.0, "noisy f0 {noisy_f0}");
}

#[test]
fn aperiodicity_is_reported_exactly_where_f0_is() {
    let rate = 16_000u32;
    let samples: Vec<f32> = (0..rate as usize)
        .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / rate as f64).sin() as f32)
        .collect();
    let track = estimate_pitch(&samples, rate, PitchOptions::default()).unwrap();

    assert_eq!(track.aperiodicity.len(), track.f0.len());
    for (i, (f, a)) in track.f0.iter().zip(track.aperiodicity.iter()).enumerate() {
        assert_eq!(
            f.is_some(),
            a.is_some(),
            "frame {i}: f0 and aperiodicity must agree on voicing"
        );
        if let Some(a) = a {
            assert!(
                (0.0..=1.0).contains(a),
                "frame {i}: aperiodicity {a} out of range"
            );
        }
    }
}

#[test]
fn silence_reports_no_hnr_at_all() {
    let rate = 16_000u32;
    let track = estimate_pitch(&vec![0.0f32; rate as usize], rate, PitchOptions::default())
        .expect("silence must not error");
    assert_eq!(track.voiced_fraction, 0.0);
    assert!(track.mean_hnr_db.is_none(), "silence has no HNR to report");
    assert!(track.aperiodicity.iter().all(|a| a.is_none()));
}

#[test]
fn hnr_is_monotonic_in_aperiodicity_and_finite_at_the_ends() {
    // The clamp exists so averaging cannot be poisoned by a single +/-inf.
    assert!(hnr_db(0.0).is_finite());
    assert!(hnr_db(1.0).is_finite());
    assert!(hnr_db(0.0) > hnr_db(0.1));
    assert!(hnr_db(0.1) > hnr_db(0.5));
    assert!(hnr_db(0.5) > hnr_db(0.9));
    // a = 0.5 is equal harmonic and noise power, so 0 dB by definition.
    assert!(
        hnr_db(0.5).abs() < 1e-9,
        "a=0.5 must be 0 dB, got {}",
        hnr_db(0.5)
    );
}
