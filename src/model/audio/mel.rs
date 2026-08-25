//! Mel spectrogram computation for audio preprocessing.
//!
//! Pure CPU computation that produces a `Vec<f32>` in `[num_mel_bins, num_frames]` layout.
//! The caller constructs a `Tensor` on the appropriate device from the result.
//!
//! [`MelOptions::whisper`] reproduces HuggingFace's `WhisperFeatureExtractor`
//! exactly: pad-or-trim to 30 s, reflect-pad by `n_fft / 2`, periodic Hann,
//! Slaney mel scale with Slaney filter normalization, and Whisper's clamped
//! log10 compression. [`MelOptions::new`] keeps the older generic behavior
//! (HTK scale, unnormalized filters, natural log, no centering, no padding).

use std::f32::consts::PI;

use super::reflection_pad::reflection_pad_1d;
use crate::error::{Error, Result};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::Complex64;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Which Hz↔mel warping to use when placing the filterbank edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// HTK formula: `2595 * log10(1 + hz / 700)`. librosa's `htk=True`.
    Htk,
    /// Slaney formula: linear below 1 kHz, logarithmic above. librosa's
    /// `htk=False`, and what Whisper uses.
    Slaney,
}

impl MelScale {
    /// Hz to mel under this warping.
    #[inline]
    pub fn to_mel(self, hz: f64) -> f64 {
        match self {
            Self::Htk => hz_to_mel_htk64(hz),
            Self::Slaney => hz_to_mel_slaney(hz),
        }
    }

    /// Mel back to Hz under this warping.
    #[inline]
    pub fn to_hz(self, mel: f64) -> f64 {
        match self {
            Self::Htk => mel_to_hz_htk64(mel),
            Self::Slaney => mel_to_hz_slaney(mel),
        }
    }
}

/// Post-construction scaling applied to each triangular filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    /// Leave the unit-peak triangles as built.
    None,
    /// Scale filter `m` by `2 / (edge[m + 2] - edge[m])` so each filter has
    /// approximately unit area. librosa's `norm="slaney"`.
    Slaney,
}

/// How mel energies are compressed to the returned values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSpec {
    /// `ln(max(energy, 1e-10))`.
    Natural,
    /// `log10(max(energy, 1e-10))`, floored at `global_max - 8`, then
    /// rescaled by `(x + 4) / 4`. The floor uses ONE maximum over the whole
    /// spectrogram, not a per-frame maximum.
    Whisper,
}

/// Every parameter of the mel front end.
#[derive(Debug, Clone, PartialEq)]
pub struct MelOptions {
    /// FFT length. Any value >= 1; numr's CPU FFT handles non-powers of two.
    pub n_fft: usize,
    /// Samples between consecutive frame starts.
    pub hop_length: usize,
    /// Length of the Hann window. Must be <= `n_fft`; the remainder of each
    /// frame is zero-padded.
    pub win_length: usize,
    /// Number of mel filterbank channels.
    pub num_mel_bins: usize,
    /// Lowest filterbank edge frequency in Hz.
    pub fmin: f32,
    /// Highest filterbank edge frequency in Hz. `None` means `sample_rate / 2`.
    pub fmax: Option<f32>,
    /// Hz↔mel warping.
    pub mel_scale: MelScale,
    /// Filter normalization.
    pub normalize: MelNorm,
    /// Log compression.
    pub log: LogSpec,
    /// Reflect-pad the signal by `n_fft / 2` on both ends before framing, and
    /// drop the final frame afterwards (PyTorch `center=True` semantics).
    pub center: bool,
    /// Zero-fill or truncate the signal to exactly this many samples before
    /// any padding or framing. `None` leaves the signal as given.
    pub pad_to_samples: Option<usize>,
}

impl MelOptions {
    /// Generic 25 ms / 10 ms front end at 16 kHz rates: HTK scale,
    /// unnormalized filters, natural log, no centering, no pad-or-trim.
    ///
    /// This is what [`compute_mel_spectrogram`] uses.
    pub fn new(num_mel_bins: usize) -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            num_mel_bins,
            fmin: 0.0,
            fmax: None,
            mel_scale: MelScale::Htk,
            normalize: MelNorm::None,
            log: LogSpec::Natural,
            center: false,
            pad_to_samples: None,
        }
    }

    /// Whisper's preprocessing, matching HuggingFace's `WhisperFeatureExtractor`.
    ///
    /// `num_mel_bins` is 80 for every Whisper checkpoint except large-v3,
    /// which uses 128. The 30 s pad-or-trim is what makes every output
    /// exactly 3000 frames regardless of input length.
    pub fn whisper(num_mel_bins: usize, sample_rate: usize) -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            num_mel_bins,
            fmin: 0.0,
            fmax: None,
            mel_scale: MelScale::Slaney,
            normalize: MelNorm::Slaney,
            log: LogSpec::Whisper,
            center: true,
            pad_to_samples: Some(30 * sample_rate),
        }
    }
}

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

/// Slope of the linear part of the Slaney scale: 1 mel per `200/3` Hz.
const SLANEY_F_SP: f64 = 200.0 / 3.0;
/// Frequency at which the Slaney scale switches from linear to logarithmic.
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
/// The mel value of [`SLANEY_MIN_LOG_HZ`]; exactly 15.
const SLANEY_MIN_LOG_MEL: f64 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;

/// Natural-log step per mel above 1 kHz: 6.4x in frequency over 27 mels.
#[inline]
fn slaney_logstep() -> f64 {
    6.4f64.ln() / 27.0
}

/// Convert Hz to mel on the Slaney scale (librosa's `htk=False`).
#[inline]
pub fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz >= SLANEY_MIN_LOG_HZ {
        SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / slaney_logstep()
    } else {
        hz / SLANEY_F_SP
    }
}

/// Convert a Slaney-scale mel value back to Hz.
#[inline]
pub fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel >= SLANEY_MIN_LOG_MEL {
        SLANEY_MIN_LOG_HZ * (slaney_logstep() * (mel - SLANEY_MIN_LOG_MEL)).exp()
    } else {
        SLANEY_F_SP * mel
    }
}

/// Convert Hz to mel on the HTK scale, in f64.
#[inline]
fn hz_to_mel_htk64(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert an HTK mel value back to Hz, in f64.
#[inline]
fn mel_to_hz_htk64(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
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

/// Compute `num_mel_bins + 2` filterbank edge frequencies in Hz, evenly
/// spaced on the chosen mel scale.
///
/// Done in f64: the Slaney edges go through `ln`/`exp`, and the filter
/// normalization then divides by small differences between adjacent edges,
/// where f32 rounding is visible in the resulting energies.
pub fn mel_frequencies_with(
    num_mel_bins: usize,
    fmin: f64,
    fmax: f64,
    scale: MelScale,
) -> Vec<f64> {
    let mel_min = scale.to_mel(fmin);
    let mel_max = scale.to_mel(fmax);
    let n = num_mel_bins + 2;
    (0..n)
        .map(|i| scale.to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n - 1) as f64))
        .collect()
}

/// Build the `[num_mel_bins, n_fft / 2 + 1]` triangular filterbank, row-major.
///
/// Uses librosa's ramp formulation so the shared edges between neighbouring
/// filters agree exactly rather than by a floating-point coincidence.
fn mel_filterbank(
    num_mel_bins: usize,
    n_fft: usize,
    sample_rate: f64,
    fmin: f64,
    fmax: f64,
    scale: MelScale,
    normalize: MelNorm,
) -> Vec<f64> {
    let num_fft_bins = n_fft / 2 + 1;
    let edges = mel_frequencies_with(num_mel_bins, fmin, fmax, scale);
    let fft_freqs: Vec<f64> = (0..num_fft_bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();

    let mut filterbank = vec![0.0f64; num_mel_bins * num_fft_bins];
    for m in 0..num_mel_bins {
        let lower_width = edges[m + 1] - edges[m];
        let upper_width = edges[m + 2] - edges[m + 1];
        for k in 0..num_fft_bins {
            let freq = fft_freqs[k];
            // Rising edge of filter m, and falling edge of the same filter.
            let lower = if lower_width > 0.0 {
                (freq - edges[m]) / lower_width
            } else {
                0.0
            };
            let upper = if upper_width > 0.0 {
                (edges[m + 2] - freq) / upper_width
            } else {
                0.0
            };
            filterbank[m * num_fft_bins + k] = lower.min(upper).max(0.0);
        }
        if normalize == MelNorm::Slaney {
            let span = edges[m + 2] - edges[m];
            if span > 0.0 {
                let enorm = 2.0 / span;
                for k in 0..num_fft_bins {
                    filterbank[m * num_fft_bins + k] *= enorm;
                }
            }
        }
    }
    filterbank
}

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
mod tests;
