//! Mel spectrogram computation for audio preprocessing.
//!
//! Pure CPU computation that produces a `Vec<f32>` in `[num_mel_bins, num_frames]` layout.
//! The caller constructs a `Tensor` on the appropriate device from the result.

use std::f32::consts::PI;

use crate::error::{Error, Result};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::Complex64;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Convert frequency in Hz to mel scale (HTK formula).
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale value back to Hz.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Compute `num_mel_bins + 2` linearly spaced mel frequencies, converted back to Hz.
pub fn mel_frequencies(num_mel_bins: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let n = num_mel_bins + 2;
    (0..n)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n - 1) as f32))
        .collect()
}

/// Compute a log-mel spectrogram from raw audio samples.
///
/// Returns a `Vec<f32>` in `[num_mel_bins, num_frames]` row-major layout.
///
/// Fallible because the FFT is numr's, and numr's `rfft` reports shape/dtype
/// errors rather than panicking. Those are unreachable for the shapes built
/// here, but a wrong spectrum is silently wrong AUDIO — the caller must see the
/// error rather than receive plausible-looking silence.
///
/// Parameters:
/// - `samples`: mono 16-bit PCM as f32 (range [-1, 1])
/// - `num_mel_bins`: number of mel filterbank channels (typically 80 or 128)
/// - `sample_rate`: audio sample rate in Hz (typically 16000)
pub fn compute_mel_spectrogram(
    samples: &[f32],
    num_mel_bins: usize,
    sample_rate: usize,
) -> Result<Vec<f32>> {
    let window_size = 400; // 25ms at 16kHz
    let hop_size = 160; // 10ms at 16kHz
    // Pad to next power of 2, numr's `rfft` size requirement. The frequency
    // resolution shift (vs n_fft=400) is smoothed out by the mel filterbank.
    let fft_size = 512;

    // Precompute Hann window
    let hann: Vec<f32> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / window_size as f32).cos()))
        .collect();

    // Number of frames
    let num_frames = if samples.len() >= window_size {
        (samples.len() - window_size) / hop_size + 1
    } else {
        0
    };

    let num_fft_bins = fft_size / 2 + 1;

    // Compute mel filterbank: [num_mel_bins, num_fft_bins]
    let fmax = sample_rate as f32 / 2.0;
    let mel_freqs = mel_frequencies(num_mel_bins, 0.0, fmax);
    let fft_freqs: Vec<f32> = (0..num_fft_bins)
        .map(|i| i as f32 * sample_rate as f32 / fft_size as f32)
        .collect();

    let mut filterbank = vec![0.0f32; num_mel_bins * num_fft_bins];
    for m in 0..num_mel_bins {
        let f_left = mel_freqs[m];
        let f_center = mel_freqs[m + 1];
        let f_right = mel_freqs[m + 2];
        for k in 0..num_fft_bins {
            let freq = fft_freqs[k];
            let weight = if freq >= f_left && freq <= f_center && f_center > f_left {
                (freq - f_left) / (f_center - f_left)
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
            filterbank[m * num_fft_bins + k] = weight;
        }
    }

    let mut output = vec![0.0f32; num_mel_bins * num_frames];
    if num_frames == 0 {
        // Nothing to window or FFT — also sidesteps a zero-batch shape into
        // numr's rfft, which is not exercised elsewhere in this crate.
        return Ok(output);
    }

    // Window ALL frames into one `[num_frames, fft_size]` buffer, zero-padded
    // beyond `window_size`, so the FFT below runs as a single batched call
    // instead of one call per frame.
    let mut windowed = vec![0.0f32; num_frames * fft_size];
    for (frame_idx, buf) in windowed.chunks_mut(fft_size).enumerate() {
        let start = frame_idx * hop_size;

        // Apply Hann window over the first `window_size` samples; the rest stays
        // zero (implicit zero-padding up to `fft_size`).
        for n in 0..window_size {
            let sample = if start + n < samples.len() {
                samples[start + n]
            } else {
                0.0
            };
            buf[n] = sample * hann[n];
        }
    }

    // Batched real FFT → power spectrum, `[num_frames, num_fft_bins]`. numr owns
    // the FFT (boostr does not reimplement it) — see `power_spectra` below.
    let power = power_spectra(&windowed, num_frames, fft_size)?;

    for (frame_idx, frame_power) in power.chunks(num_fft_bins).take(num_frames).enumerate() {
        // Apply mel filterbank and log.
        for m in 0..num_mel_bins {
            let mut energy = 0.0f32;
            for k in 0..num_fft_bins {
                energy += filterbank[m * num_fft_bins + k] * frame_power[k];
            }
            output[m * num_frames + frame_idx] = energy.max(1e-10).ln();
        }
    }

    Ok(output)
}

/// Batched real FFT over `[num_frames, fft_size]`, returning `|X[k]|^2` as
/// `[num_frames, fft_size/2 + 1]`.
///
/// numr owns the FFT; boostr does not reimplement it.
///
/// The error paths (shape/dtype/size validation) are unreachable given
/// `windowed.len() == num_frames * fft_size` and a fixed power-of-two
/// `fft_size`. They are still propagated rather than swallowed: substituting an
/// all-zero spectrum would turn an impossible error into silently wrong audio
/// features, which is far harder to diagnose than a returned error.
fn power_spectra(windowed: &[f32], num_frames: usize, fft_size: usize) -> Result<Vec<f32>> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::try_from_slice(windowed, &[num_frames, fft_size], &device)
        .map_err(Error::Numr)?;
    let spectrum = client
        .rfft(&input, FftNormalization::None)
        .map_err(Error::Numr)?
        .contiguous()
        .map_err(Error::Numr)?;
    let bins: Vec<Complex64> = spectrum.to_vec();
    Ok(bins.iter().map(|c| c.re * c.re + c.im * c.im).collect())
}

#[cfg(test)]
mod tests {
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
}
