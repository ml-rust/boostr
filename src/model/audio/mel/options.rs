//! Every parameter of the mel front end.

use super::filterbank::MelNorm;
use super::scale::MelScale;

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
    /// This is what [`compute_mel_spectrogram`](super::compute_mel_spectrogram) uses.
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
