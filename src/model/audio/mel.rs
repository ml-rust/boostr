//! Mel spectrogram computation for audio preprocessing.
//!
//! Pure CPU computation that produces a `Vec<f32>` in `[num_mel_bins, num_frames]` layout.
//! The caller constructs a `Tensor` on the appropriate device from the result.

use std::f32::consts::PI;

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
/// Parameters:
/// - `samples`: mono 16-bit PCM as f32 (range [-1, 1])
/// - `num_mel_bins`: number of mel filterbank channels (typically 80 or 128)
/// - `sample_rate`: audio sample rate in Hz (typically 16000)
pub fn compute_mel_spectrogram(
    samples: &[f32],
    num_mel_bins: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let window_size = 400; // 25ms at 16kHz
    let hop_size = 160; // 10ms at 16kHz
    let fft_size = window_size; // no zero-padding beyond window

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

    // Compute STFT power spectrum and apply filterbank
    let mut output = vec![0.0f32; num_mel_bins * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;

        // Apply window and compute DFT (real input, brute-force DFT for correctness)
        let mut real_parts = vec![0.0f32; num_fft_bins];
        let mut imag_parts = vec![0.0f32; num_fft_bins];

        for k in 0..num_fft_bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for n in 0..window_size {
                let sample = if start + n < samples.len() {
                    samples[start + n]
                } else {
                    0.0
                };
                let windowed = sample * hann[n];
                let angle = -2.0 * PI * k as f32 * n as f32 / fft_size as f32;
                re += windowed * angle.cos();
                im += windowed * angle.sin();
            }
            real_parts[k] = re;
            imag_parts[k] = im;
        }

        // Power spectrum
        let power: Vec<f32> = real_parts
            .iter()
            .zip(imag_parts.iter())
            .map(|(re, im)| re * re + im * im)
            .collect();

        // Apply mel filterbank and log
        for m in 0..num_mel_bins {
            let mut energy = 0.0f32;
            for k in 0..num_fft_bins {
                energy += filterbank[m * num_fft_bins + k] * power[k];
            }
            // Log with floor to avoid log(0)
            output[m * num_frames + frame_idx] = (energy.max(1e-10)).ln();
        }
    }

    output
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
        let result = compute_mel_spectrogram(&samples, 128, 16000);
        let num_frames = (16000 - 400) / 160 + 1; // 98
        assert_eq!(result.len(), 128 * num_frames);
    }

    #[test]
    fn test_spectrogram_short_audio() {
        // Too short for even one frame
        let samples = vec![0.0f32; 100];
        let result = compute_mel_spectrogram(&samples, 80, 16000);
        assert!(result.is_empty());
    }
}
