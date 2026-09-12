//! Framing, batched FFT, mel projection and log compression.

use std::f32::consts::PI;

use super::filterbank::mel_filterbank;
use super::options::{LogSpec, MelOptions};
use crate::error::{Error, Result};
use crate::model::audio::reflection_pad::reflection_pad_1d;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::Complex64;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Compute a log-mel spectrogram from raw audio samples.
///
/// Returns a `Vec<f32>` in `[num_mel_bins, num_frames]` row-major layout.
///
/// Thin wrapper over [`compute_mel_spectrogram_with`] using [`MelOptions::new`]:
/// a 400-sample window, 160-sample hop, HTK mel scale, unnormalized filters
/// and a natural log. It does NOT produce Whisper-compatible features — use
/// [`MelOptions::whisper`] with [`compute_mel_spectrogram_with`] for that.
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
    compute_mel_spectrogram_with(samples, sample_rate, &MelOptions::new(num_mel_bins))
}

/// Compute a log-mel spectrogram under an explicit [`MelOptions`].
///
/// Returns a `Vec<f32>` in `[num_mel_bins, num_frames]` row-major layout.
///
/// Fallible because the FFT is numr's, and numr's `rfft` reports shape/dtype
/// errors rather than panicking. A wrong spectrum is silently wrong AUDIO —
/// the caller must see the error rather than receive plausible-looking silence.
pub fn compute_mel_spectrogram_with(
    samples: &[f32],
    sample_rate: usize,
    opts: &MelOptions,
) -> Result<Vec<f32>> {
    if opts.n_fft == 0 || opts.hop_length == 0 || opts.win_length == 0 {
        return Err(Error::InvalidArgument {
            arg: "opts",
            reason: format!(
                "n_fft ({}), hop_length ({}) and win_length ({}) must all be non-zero",
                opts.n_fft, opts.hop_length, opts.win_length
            ),
        });
    }
    if opts.win_length > opts.n_fft {
        return Err(Error::InvalidArgument {
            arg: "opts.win_length",
            reason: format!(
                "win_length ({}) must be <= n_fft ({})",
                opts.win_length, opts.n_fft
            ),
        });
    }
    if opts.num_mel_bins == 0 {
        return Err(Error::InvalidArgument {
            arg: "opts.num_mel_bins",
            reason: "must be non-zero".to_string(),
        });
    }
    if sample_rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "sample_rate",
            reason: "must be non-zero".to_string(),
        });
    }
    let nyquist = sample_rate as f32 / 2.0;
    if let Some(explicit_fmax) = opts.fmax
        && explicit_fmax > nyquist
    {
        return Err(Error::InvalidArgument {
            arg: "opts.fmax",
            reason: format!(
                "fmax ({explicit_fmax}) must not exceed the Nyquist frequency ({nyquist}) for sample_rate {sample_rate}"
            ),
        });
    }
    let fmax = opts.fmax.unwrap_or(nyquist);
    if opts.fmin >= fmax {
        return Err(Error::InvalidArgument {
            arg: "opts.fmin",
            reason: format!("fmin ({}) must be less than fmax ({fmax})", opts.fmin),
        });
    }

    let num_mel_bins = opts.num_mel_bins;
    let n_fft = opts.n_fft;
    let hop = opts.hop_length;
    let win = opts.win_length;

    // 1. Pad or trim to a fixed length. Whisper's 30 s window: shorter clips
    //    are zero-filled, longer ones truncated, so every output has the same
    //    frame count.
    let signal: Vec<f32> = match opts.pad_to_samples {
        Some(target) => {
            let mut v = vec![0.0f32; target];
            let n = samples.len().min(target);
            v[..n].copy_from_slice(&samples[..n]);
            v
        }
        None => samples.to_vec(),
    };

    // 2. Reflect-pad both ends by n_fft / 2, PyTorch `center=True` semantics.
    //    Reuses the crate's `reflection_pad_1d` rather than open-coding a
    //    second mirror.
    let signal = if opts.center && n_fft / 2 > 0 {
        let pad = n_fft / 2;
        if signal.len() <= pad {
            return Err(Error::InvalidArgument {
                arg: "samples",
                reason: format!(
                    "centered framing needs more than n_fft/2 ({pad}) samples, got {}",
                    signal.len()
                ),
            });
        }
        let device = CpuDevice::new();
        let as_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[1, 1, signal.len()], &device)
            .map_err(Error::Numr)?;
        reflection_pad_1d(&as_tensor, pad, pad)?.to_vec()
    } else {
        signal
    };

    // 3. Frame count. Centering produces one frame past the reference's
    //    output (3001 vs 3000 for Whisper); the reference drops the last.
    let mut num_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop + 1
    } else {
        0
    };
    if opts.center {
        num_frames = num_frames.saturating_sub(1);
    }

    let num_fft_bins = n_fft / 2 + 1;

    let mut output = vec![0.0f32; num_mel_bins * num_frames];
    if num_frames == 0 {
        // Nothing to window or FFT — also sidesteps a zero-batch shape into
        // numr's rfft, which is not exercised elsewhere in this crate. The
        // filterbank isn't needed either, so it's never built.
        return Ok(output);
    }

    let filterbank = mel_filterbank(
        num_mel_bins,
        n_fft,
        sample_rate as f64,
        opts.fmin as f64,
        fmax as f64,
        opts.mel_scale,
        opts.normalize,
    );

    // 4. Periodic Hann over `win` samples, zero-padded out to `n_fft`.
    let hann: Vec<f32> = (0..win)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / win as f32).cos()))
        .collect();

    // Window ALL frames into one `[num_frames, n_fft]` buffer so the FFT below
    // runs as a single batched call instead of one call per frame.
    let mut windowed = vec![0.0f32; num_frames * n_fft];
    for (frame_idx, buf) in windowed.chunks_mut(n_fft).enumerate() {
        let start = frame_idx * hop;
        for n in 0..win {
            let sample = signal.get(start + n).copied().unwrap_or(0.0);
            buf[n] = sample * hann[n];
        }
    }

    // 5. Batched real FFT → power spectrum, `[num_frames, num_fft_bins]`.
    //    numr owns the FFT (boostr does not reimplement it).
    let power = power_spectra(&windowed, num_frames, n_fft)?;

    // 6 + 7. Mel projection then log compression.
    let floor = 1e-10f64;
    for (frame_idx, frame_power) in power.chunks(num_fft_bins).take(num_frames).enumerate() {
        for m in 0..num_mel_bins {
            let mut energy = 0.0f64;
            for k in 0..num_fft_bins {
                energy += filterbank[m * num_fft_bins + k] * frame_power[k] as f64;
            }
            let compressed = match opts.log {
                LogSpec::Natural => energy.max(floor).ln(),
                LogSpec::Whisper => energy.max(floor).log10(),
            };
            output[m * num_frames + frame_idx] = compressed as f32;
        }
    }

    if opts.log == LogSpec::Whisper {
        // ONE maximum over every bin and every frame — a per-frame maximum
        // would renormalize silence up to the level of speech.
        let global_max = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let clamp_floor = global_max - 8.0;
        for v in output.iter_mut() {
            *v = (v.max(clamp_floor) + 4.0) / 4.0;
        }
    }

    Ok(output)
}

/// Batched real FFT over `[num_frames, n_fft]`, returning `|X[k]|^2` as
/// `[num_frames, n_fft/2 + 1]`.
///
/// numr owns the FFT; boostr does not reimplement it. `n_fft` need not be a
/// power of two — numr's CPU FFT falls back to Bluestein.
///
/// The error paths (shape/dtype/size validation) are unreachable given
/// `windowed.len() == num_frames * n_fft`. They are still propagated rather
/// than swallowed: substituting an all-zero spectrum would turn an impossible
/// error into silently wrong audio features, which is far harder to diagnose
/// than a returned error.
fn power_spectra(windowed: &[f32], num_frames: usize, n_fft: usize) -> Result<Vec<f32>> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(windowed, &[num_frames, n_fft], &device)
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
}
