//! DSP behaviour tests for the polyphase resampler.
//!
//! Frequency content is measured with a single-bin DFT written here, so a bug in
//! `stft` or any other boostr code cannot mask a bug in the resampler. Every
//! analysis window holds a whole number of cycles of the tone under test, which
//! removes scalloping loss and lets the assertions pin real amplitudes.

use super::*;
use crate::model::audio::wav_decode::WavData;

use std::f64::consts::TAU;

/// Unit-amplitude sine of `freq` Hz sampled at `rate`.
fn tone(freq: f64, rate: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| (TAU * freq * n as f64 / rate as f64).sin() as f32)
        .collect()
}

/// Amplitude of the `freq` component of `x`, by direct single-bin DFT.
///
/// Returns `2 |X(freq)| / N`, which is the peak amplitude of a real sinusoid
/// when the window spans a whole number of its cycles.
fn amplitude_at(x: &[f32], rate: u32, freq: f64) -> f64 {
    let w = TAU * freq / rate as f64;
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (n, &v) in x.iter().enumerate() {
        let phase = w * n as f64;
        re += v as f64 * phase.cos();
        im -= v as f64 * phase.sin();
    }
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}

fn rms(x: &[f32]) -> f64 {
    let sum: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sum / x.len() as f64).sqrt()
}

fn db(value: f64, reference: f64) -> f64 {
    20.0 * (value / reference).log10()
}

/// Catches a resampler that shifts pitch (wrong phase step or wrong anchor) or
/// loses level (missing the `up` gain that zero stuffing costs).
#[test]
fn tone_survives_downsampling_48k_to_16k() {
    let input = tone(1000.0, 48_000, 96_000);
    let out = resample(&input, 48_000, 16_000).expect("resample");
    assert_eq!(out.len(), 32_000);

    // One second, clear of both edge ramps; 1 kHz is exactly 1000 cycles.
    let window = &out[8_000..24_000];
    let amp = amplitude_at(window, 16_000, 1000.0);
    assert!(amp > 0.97 && amp < 1.03, "1 kHz amplitude {amp}");
    let level = rms(window);
    assert!(
        level > 0.686 && level < 0.728,
        "rms {level}, expected about 0.707"
    );
    // No energy anywhere else worth speaking of.
    let spur = amplitude_at(window, 16_000, 3000.0);
    assert!(spur < 0.01, "3 kHz spur {spur}");
}

/// THE alias rejection test. A naive decimator that drops samples without
/// filtering folds 15 kHz down to |15000 - 16000| = 1 kHz at nearly full
/// amplitude. A correct anti-aliased resampler stops it, while leaving 7 kHz —
/// below the 8 kHz output Nyquist — intact.
#[test]
fn above_nyquist_content_is_rejected_not_folded() {
    // 7 kHz is inside the passband, so it must survive.
    let pass = resample(&tone(7000.0, 48_000, 96_000), 48_000, 16_000).expect("resample 7 kHz");
    let pass_window = &pass[8_000..24_000];
    let kept = amplitude_at(pass_window, 16_000, 7000.0);
    assert!(kept > 0.5, "7 kHz amplitude {kept}, expected near 1.0");

    // 15 kHz is above it, so it must be attenuated rather than folded to 1 kHz.
    let stop = resample(&tone(15_000.0, 48_000, 96_000), 48_000, 16_000).expect("resample 15 kHz");
    let stop_window = &stop[8_000..24_000];
    let folded = amplitude_at(stop_window, 16_000, 1000.0);
    let rejection = db(folded, kept);
    assert!(
        rejection < -40.0,
        "alias at 1 kHz is {folded} ({rejection:.1} dB relative to the kept 7 kHz tone \
         at {kept}); a naive decimator scores about 0 dB"
    );
    // Nothing else survives either: the whole 15 kHz tone is gone.
    let leftover = rms(stop_window);
    assert!(leftover < 0.02, "stopband leakage rms {leftover}");
}

/// Catches an upsampler that leaves the spectral images zero stuffing creates —
/// the 1 kHz tone mirrored about the old 16 kHz rate, at 15 kHz.
#[test]
fn upsampling_16k_to_48k_preserves_tone_without_images() {
    let input = tone(1000.0, 16_000, 32_000);
    let out = resample(&input, 16_000, 48_000).expect("resample");
    assert_eq!(out.len(), 96_000);

    let window = &out[24_000..72_000];
    let amp = amplitude_at(window, 48_000, 1000.0);
    assert!(amp > 0.97 && amp < 1.03, "1 kHz amplitude {amp}");
    let image = amplitude_at(window, 48_000, 15_000.0);
    assert!(
        db(image, amp) < -40.0,
        "image at 15 kHz is {image} against {amp}"
    );
    let image_high = amplitude_at(window, 48_000, 17_000.0);
    assert!(image_high < 0.01, "image at 17 kHz {image_high}");
}

/// Catches a resampler that filters when it has nothing to do, which would both
/// waste work and smear a signal that should pass through untouched.
#[test]
fn equal_rates_return_the_input_bit_identically() {
    let input: Vec<f32> = (0..1000).map(|n| (n as f32 * 0.001).sin() * 0.37).collect();
    let out = resample(&input, 44_100, 44_100).expect("resample");
    assert_eq!(out, input);
}

/// Catches a wrapping `usize` product in the length formula, and off-by-one in
/// the ceiling.
#[test]
fn output_length_is_the_documented_ceiling() {
    let exact = resample(&vec![0.0; 44_100], 44_100, 16_000).expect("resample");
    assert_eq!(exact.len(), 16_000);

    // 1000 * 16000 / 44100 = 362.81..., so 363.
    let rounded = resample(&vec![0.0; 1000], 44_100, 16_000).expect("resample");
    assert_eq!(rounded.len(), 363);

    let empty = resample(&[], 44_100, 16_000).expect("resample");
    assert!(empty.is_empty());
}

/// Catches a converter that only handles integer ratios. 44100 to 16000 reduces
/// to up = 160, down = 441 — 160 distinct polyphase branches, none of them a
/// whole number of input samples apart.
#[test]
fn coprime_ratio_44100_to_16000_filters_correctly() {
    let out = resample(&tone(1000.0, 44_100, 88_200), 44_100, 16_000).expect("resample");
    assert_eq!(out.len(), 32_000);
    let window = &out[8_000..24_000];
    let amp = amplitude_at(window, 16_000, 1000.0);
    assert!(amp > 0.97 && amp < 1.03, "1 kHz amplitude {amp}");

    // 12 kHz is above the 8 kHz output Nyquist; unfiltered it would fold to
    // |12000 - 16000| = 4 kHz.
    let stop = resample(&tone(12_000.0, 44_100, 88_200), 44_100, 16_000).expect("resample");
    let folded = amplitude_at(&stop[8_000..24_000], 16_000, 4000.0);
    assert!(db(folded, amp) < -40.0, "alias at 4 kHz is {folded}");
}

/// Catches panicking index arithmetic when the input is shorter than the filter
/// half-width, so nearly every tap reads out of range.
#[test]
fn very_short_inputs_do_not_panic() {
    let down = resample(&[0.5, -0.25, 0.75], 48_000, 16_000).expect("resample");
    assert_eq!(down.len(), 1);
    assert!(down[0].is_finite());

    let up = resample(&[1.0], 16_000, 48_000).expect("resample");
    assert_eq!(up.len(), 3);
    assert!(up.iter().all(|v| v.is_finite()));

    let one_to_none = resample(&[1.0, -1.0], 44_100, 16_000).expect("resample");
    assert_eq!(one_to_none.len(), 1);
}

/// Catches a second, divergent downmix inside `to_mono_at_rate`.
#[test]
fn to_mono_at_rate_matches_to_mono_then_resample() {
    let left = tone(500.0, 48_000, 4800);
    let right = tone(1500.0, 48_000, 4800);
    let mut samples = Vec::with_capacity(9600);
    for (l, r) in left.iter().zip(right.iter()) {
        samples.push(*l);
        samples.push(*r);
    }
    let data = WavData {
        samples,
        sample_rate: 48_000,
        channels: 2,
    };

    let combined = to_mono_at_rate(&data, 16_000).expect("to_mono_at_rate");
    let mono = to_mono(&data.samples, data.channels).expect("to_mono");
    let expected = resample(&mono, 48_000, 16_000).expect("resample");
    assert_eq!(combined, expected);

    // Same rate short-circuits to the plain downmix.
    let same = to_mono_at_rate(&data, 48_000).expect("to_mono_at_rate");
    assert_eq!(same, mono);
}

/// Catches silently accepting a nonsense rate, and unbounded filter allocation
/// for a ratio that barely reduces.
#[test]
fn invalid_rates_and_oversized_filters_are_rejected() {
    assert!(resample(&[0.0; 4], 0, 16_000).is_err());
    assert!(resample(&[0.0; 4], 48_000, 0).is_err());
    assert!(resample_with_taps(&[0.0; 4], 48_000, 16_000, 0).is_err());

    // 44100 and 16001 are coprime: up = 16001, down = 44100, far past the cap.
    let err = resample(&[0.0; 4], 44_100, 16_001).expect_err("filter is too large");
    let text = format!("{err}");
    assert!(text.contains("44100"), "{text}");
    assert!(text.contains("16001"), "{text}");
}
