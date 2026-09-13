//! Multi-resolution spectral distance and two-signal SNR, for comparing a
//! decoded/rendered take against its reference (e.g. an `AudioVAE` decoded at
//! F16 vs the same latent decoded at F32).
//!
//! [`TakeQuality::snr_db`](crate::quality::TakeQuality) is a SINGLE-signal
//! measure (RMS against that same signal's own noise floor). [`snr_db`] here
//! is a TWO-signal measure (one signal against a reference), a different
//! question with the same name in DSP practice — this module keeps its own
//! name rather than overloading the field.
//!
//! **CPU-only.** Each STFT here runs on a few seconds of audio at three fixed
//! resolutions; there is no batching to keep this work on a GPU client for.

use crate::error::{Error, Result};
use boostr::model::audio::stft::{StftOptions, stft};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// FFT sizes the multi-resolution loss averages over: short window for
/// transient timing, long window for harmonic/formant detail. Standard trio
/// from multi-resolution STFT losses in vocoder literature (e.g. UnivNet,
/// HiFi-GAN variants).
const FFT_SIZES: [usize; 3] = [512, 1024, 2048];

/// Absolute magnitude floor added inside every `log`, used only as a
/// fallback when a resolution's reference peak is itself 0 (a fully silent
/// reference has no scale for [`REL_LOG_EPS`] to be relative to). Also floors
/// the spectral-convergence denominator in that same degenerate case.
const LOG_EPS: f64 = 1e-10;

/// Per-resolution log floor, relative to that resolution's own reference peak
/// magnitude (`peak_a`) — see [`log_floor`].
///
/// # Why relative to the peak, and why F64 for the transform
///
/// In EXACT arithmetic the STFT is linear, so for `b = a * s` the magnitude
/// at every bin satisfies `mag_b == s * mag_a` exactly, and `log` is
/// scale-invariant for any nonzero value: `ln(mag_a*s + 0) - ln(mag_a + 0) ==
/// ln(s)`, at every bin down to the true zero, however small `mag_a` is.
/// Measured directly (`stft`'s underlying `rfft`, single frame, no window):
/// this holds in F64 to ~1e-14 relative error even at the smallest
/// representable nonzero magnitude.
///
/// What breaks it is NOT the transform — it is that `a`/`b` are ordinary F32
/// PCM samples, and F32 can represent `s` and each sample to only ~6e-8
/// relative precision. `b`'s samples are therefore not EXACTLY `s * a`'s
/// samples as real numbers; they carry their own independent quantization
/// noise on that order. Propagated through a windowed multi-frame STFT, that
/// per-sample noise spreads across bins as an ADDITIVE term roughly
/// independent of frequency, and dominates wherever a bin's true signal
/// (e.g. the leakage tail of a tone that does not land on an exact bin — the
/// common case: 440 Hz does not land on a bin of a 512-point transform at
/// 16 kHz) decays below it. A real measurement on such a bin: `mag_a =
/// 1.03e-8`, `mag_b = 1.29e-7` — a ratio far from the expected `s ~= 1.12`,
/// entirely explained by quantization noise, not by any defect in the
/// transform (confirmed by re-running the same comparison on `a`/`b`
/// generated directly in F64, never touching F32: log-magnitude L1 landed
/// within noise of `ln(s)`).
///
/// Running the transform itself in F64 (rather than F32) removes ITS OWN
/// rounding as a contributor (F64's floor is ~1e-15 relative, immaterial next
/// to F32 input noise at ~1e-8 relative), isolating the one remaining error
/// source to something a floor CAN address: input quantization noise, which
/// is bounded in absolute terms by roughly `peak_a * 1e-8` for these sizes.
/// [`REL_LOG_EPS`] sits comfortably above that (with margin) so bins at or
/// below it are treated as agreeing (diff ~= 0) rather than amplifying
/// sample-level rounding into a large per-bin log difference, while staying
/// far enough below any bin with genuine signal (an all-bins-loud signal
/// like white noise is entirely unaffected — see the inline tests) that the
/// genuine leakage tail's `ln(s)` contribution survives.
const REL_LOG_EPS: f64 = 3e-9;

/// The floor `multi_resolution_stft_distance` adds inside `log` (and uses to
/// floor the spectral-convergence denominator) at one resolution:
/// [`REL_LOG_EPS`] of the reference's own peak magnitude at that resolution,
/// falling back to the absolute [`LOG_EPS`] when that peak is 0.
fn log_floor(peak_a: f64) -> f64 {
    (REL_LOG_EPS * peak_a).max(LOG_EPS)
}

/// Periodic Hann window (`0.5 - 0.5·cos(2π·i/n)`) in F64.
///
/// [`boostr::model::audio::stft::hann_window`] always builds an F32 window
/// regardless of its `R` type parameter (it fills a `Vec<f32>` internally),
/// so it cannot supply the F64 window this module's F64 STFT needs — see
/// [`REL_LOG_EPS`]'s docs for why F64 matters here. Same formula, evaluated
/// in F64 throughout rather than truncated to F32 first.
fn hann_window_f64(n: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n.max(1) as f64).cos())
        .collect()
}

/// Multi-resolution STFT distance between a reference signal `a` and a test
/// signal `b`, averaged over [`FFT_SIZES`] (hop = `n_fft / 4`, periodic Hann
/// window, centered STFT). The transform itself runs in F64 — see
/// [`REL_LOG_EPS`]'s docs — though `a`/`b` are ordinary F32 samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralDistance {
    /// Mean absolute difference of `log(magnitude + `[`log_floor`]`(peak_a))`
    /// between `a` and `b`, averaged first over each resolution's bins and
    /// frames, then over [`FFT_SIZES`]. 0 for identical signals.
    pub log_mag_l1: f64,
    /// `||mag_a - mag_b||_F / max(||mag_a||_F, `[`log_floor`]`(peak_a))`,
    /// averaged over [`FFT_SIZES`]. 0 for identical signals; 1.0 when `b`
    /// shares none of `a`'s energy (e.g. `b` is silence).
    pub spectral_convergence: f64,
}

/// Compare `a` (reference) against `b` (test) at [`FFT_SIZES`] resolutions.
///
/// The two signals are compared over their common length: a longer one is
/// truncated to `min(a.len(), b.len())` rather than erroring, since a decoder
/// under test commonly returns a length off by a frame or two. A resolution
/// whose `n_fft` exceeds that common length is skipped rather than erroring,
/// so a very short clip still returns a distance over whichever resolutions
/// fit.
///
/// # Errors
/// [`Error::InvalidArgument`] when either signal is empty, `sample_rate` is
/// 0, or the common length is shorter than every entry in [`FFT_SIZES`].
pub fn multi_resolution_stft_distance(
    a: &[f32],
    b: &[f32],
    sample_rate: u32,
) -> Result<SpectralDistance> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "both signals must be non-empty, got a.len()={} b.len()={}",
                a.len(),
                b.len()
            ),
        });
    }
    if sample_rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "sample_rate",
            reason: "sample rate is 0".to_string(),
        });
    }

    let n = a.len().min(b.len());
    // Cast to F64 up front — see [`REL_LOG_EPS`]'s docs for why the transform
    // itself must run at this precision.
    let a64: Vec<f64> = a[..n].iter().map(|&s| s as f64).collect();
    let b64: Vec<f64> = b[..n].iter().map(|&s| s as f64).collect();

    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());
    let wave_a = Tensor::<CpuRuntime>::from_slice(&a64, &[1, n], &device)?;
    let wave_b = Tensor::<CpuRuntime>::from_slice(&b64, &[1, n], &device)?;

    let mut log_mag_l1_sum = 0.0f64;
    let mut spectral_convergence_sum = 0.0f64;
    let mut resolutions_used = 0usize;

    for &n_fft in &FFT_SIZES {
        if n < n_fft {
            continue;
        }
        let opts = StftOptions {
            n_fft,
            hop_length: n_fft / 4,
            center: true,
        };
        let window = Tensor::<CpuRuntime>::from_slice(&hann_window_f64(n_fft), &[n_fft], &device)?;
        let (mag_a, _) = stft(&client, &wave_a, &window, opts)?;
        let (mag_b, _) = stft(&client, &wave_b, &window, opts)?;

        let mag_a: Vec<f64> = mag_a.contiguous()?.to_vec();
        let mag_b: Vec<f64> = mag_b.contiguous()?.to_vec();
        debug_assert_eq!(mag_a.len(), mag_b.len());

        // Relative to THIS resolution's own reference peak — see
        // `log_floor`'s docs for why a fixed absolute floor is wrong here.
        let peak_a = mag_a.iter().cloned().fold(0.0f64, f64::max);
        let floor = log_floor(peak_a);

        let mut log_l1 = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&x, &y) in mag_a.iter().zip(mag_b.iter()) {
            log_l1 += ((x + floor).ln() - (y + floor).ln()).abs();
            let d = x - y;
            diff_sq += d * d;
            ref_sq += x * x;
        }
        let bins = mag_a.len().max(1) as f64;
        log_mag_l1_sum += log_l1 / bins;
        spectral_convergence_sum += diff_sq.sqrt() / ref_sq.sqrt().max(floor);
        resolutions_used += 1;
    }

    if resolutions_used == 0 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "common signal length {n} is shorter than every FFT size in {FFT_SIZES:?}"
            ),
        });
    }

    let count = resolutions_used as f64;
    Ok(SpectralDistance {
        log_mag_l1: log_mag_l1_sum / count,
        spectral_convergence: spectral_convergence_sum / count,
    })
}

/// Sample-level SNR of `test` against `reference`, in dB:
/// `10 * log10(sum(reference^2) / sum((reference - test)^2))`.
///
/// Compared over the common length, truncating a longer signal — same
/// convention as [`multi_resolution_stft_distance`]. Returns
/// [`f64::INFINITY`] for bit-identical signals (zero noise energy) and
/// [`f64::NEG_INFINITY`] when `reference` is exactly silent (zero signal
/// energy, so no ratio is defined), rather than producing `NaN` from `0/0`.
///
/// # Errors
/// [`Error::InvalidArgument`] when either signal is empty.
pub fn snr_db(reference: &[f32], test: &[f32]) -> Result<f64> {
    if reference.is_empty() || test.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "reference",
            reason: format!(
                "both signals must be non-empty, got reference.len()={} test.len()={}",
                reference.len(),
                test.len()
            ),
        });
    }
    let n = reference.len().min(test.len());
    let mut signal_energy = 0.0f64;
    let mut noise_energy = 0.0f64;
    for i in 0..n {
        let r = reference[i] as f64;
        let t = test[i] as f64;
        signal_energy += r * r;
        let d = r - t;
        noise_energy += d * d;
    }
    if noise_energy == 0.0 {
        return Ok(f64::INFINITY);
    }
    if signal_energy == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(10.0 * (signal_energy / noise_energy).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Long enough that every entry in `FFT_SIZES` (up to 2048) fits with
    /// room to spare.
    const LEN: usize = 8192;
    const RATE: u32 = 16_000;

    fn sine(amplitude: f32, freq: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                amplitude * (std::f64::consts::TAU * freq * n as f64 / RATE as f64).sin() as f32
            })
            .collect()
    }

    fn noise(len: usize, seed: u64) -> Vec<f32> {
        // xorshift64, deterministic and dependency-free: this test only needs
        // a signal with energy spread across every bin, not real randomness.
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn identical_signals_have_zero_distance() {
        let a = sine(0.8, 440.0, LEN);
        let d = multi_resolution_stft_distance(&a, &a, RATE).expect("distance");
        assert!(d.log_mag_l1 < 1e-4, "log_mag_l1 = {}", d.log_mag_l1);
        assert!(
            d.spectral_convergence < 1e-4,
            "spectral_convergence = {}",
            d.spectral_convergence
        );
    }

    #[test]
    fn identical_signals_have_infinite_snr() {
        let a = sine(0.8, 440.0, LEN);
        assert_eq!(snr_db(&a, &a).expect("snr"), f64::INFINITY);
    }

    /// A 1 dB gain scales every magnitude bin by the same factor `s =
    /// 10^(1/20)`, so in exact arithmetic `log_mag_l1` is exactly `ln(s)` at
    /// every bin, however small (`log` is scale-invariant for any nonzero
    /// value) — see [`REL_LOG_EPS`]'s docs for why an f32-sampled sine's
    /// leakage tail needs a floor to hold this to a loose tolerance, and
    /// [`pure_scale_of_white_noise_matches_ln_scale_tightly`] for the same
    /// invariant pinned tightly on a signal with no near-zero bins at all.
    #[test]
    fn one_db_scaled_copy_matches_known_log_mag_l1() {
        let a = sine(0.5, 440.0, LEN);
        let scale = 10f32.powf(1.0 / 20.0);
        let b: Vec<f32> = a.iter().map(|&s| s * scale).collect();

        let d = multi_resolution_stft_distance(&a, &b, RATE).expect("distance");
        let expected = (scale as f64).ln();
        assert!(
            (d.log_mag_l1 - expected).abs() < 0.01,
            "log_mag_l1 = {}, expected {expected}",
            d.log_mag_l1
        );
    }

    /// Same invariant as [`one_db_scaled_copy_matches_known_log_mag_l1`], but
    /// on white noise instead of a pure tone: every bin carries substantial
    /// energy (no leakage tail decaying into the F32 quantization floor), so
    /// [`log_floor`] never engages and `log_mag_l1` must land within 1e-6 of
    /// `ln(scale)` — independent of, and far tighter than, the sine test's
    /// 0.01 tolerance (which exists only to absorb that floor).
    #[test]
    fn pure_scale_of_white_noise_matches_ln_scale_tightly() {
        let a = noise(LEN, 0x1234_5678);
        let scale = 10f32.powf(1.0 / 20.0);
        let b: Vec<f32> = a.iter().map(|&s| s * scale).collect();

        let d = multi_resolution_stft_distance(&a, &b, RATE).expect("distance");
        let expected = (scale as f64).ln();
        assert!(
            (d.log_mag_l1 - expected).abs() < 1e-6,
            "log_mag_l1 = {}, expected {expected}",
            d.log_mag_l1
        );
    }

    #[test]
    fn white_noise_vs_silence_is_a_large_distance() {
        let ref_signal = sine(0.8, 440.0, LEN); // baseline scale for comparison
        let baseline = multi_resolution_stft_distance(&ref_signal, &ref_signal, RATE)
            .expect("baseline distance");

        let loud = noise(LEN, 0xA5A5_1234);
        let silence = vec![0.0f32; LEN];
        let d = multi_resolution_stft_distance(&loud, &silence, RATE).expect("distance");

        // Silence shares none of the noise's energy: the convergence ratio
        // saturates at 1.0 (the theoretical maximum for a nonnegative
        // magnitude-difference ratio against a nonzero reference).
        assert!(
            (d.spectral_convergence - 1.0).abs() < 1e-6,
            "spectral_convergence = {}",
            d.spectral_convergence
        );
        // log_mag_l1 blows up because every silent bin sits at the log floor
        // while the noisy reference bin does not — far past the near-zero
        // baseline.
        assert!(
            d.log_mag_l1 > baseline.log_mag_l1 + 1.0,
            "log_mag_l1 = {}, baseline = {}",
            d.log_mag_l1,
            baseline.log_mag_l1
        );
    }

    #[test]
    fn white_noise_vs_silence_has_very_negative_snr() {
        // reference = silence: signal_energy == 0, so SNR is defined as -inf
        // (no signal to have any ratio against), never NaN from 0/0.
        let silence = vec![0.0f32; LEN];
        let loud = noise(LEN, 42);
        assert_eq!(snr_db(&silence, &loud).expect("snr"), f64::NEG_INFINITY);
    }

    #[test]
    fn rejects_empty_signals() {
        assert!(multi_resolution_stft_distance(&[], &[1.0], RATE).is_err());
        assert!(multi_resolution_stft_distance(&[1.0], &[], RATE).is_err());
        assert!(snr_db(&[], &[1.0]).is_err());
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let a = sine(0.5, 440.0, LEN);
        assert!(multi_resolution_stft_distance(&a, &a, 0).is_err());
    }

    #[test]
    fn a_signal_shorter_than_every_fft_size_is_an_error() {
        let a = vec![0.1f32; 100];
        assert!(multi_resolution_stft_distance(&a, &a, RATE).is_err());
    }

    #[test]
    fn mismatched_lengths_are_truncated_not_rejected() {
        let a = sine(0.5, 440.0, LEN);
        let b = sine(0.5, 440.0, LEN - 37);
        assert!(multi_resolution_stft_distance(&a, &b, RATE).is_ok());
        assert!(snr_db(&a, &b).is_ok());
    }
}
