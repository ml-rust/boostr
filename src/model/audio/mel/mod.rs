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
//!
//! - `scale`: Hz↔mel warping ([`MelScale`]) and filterbank edge placement
//! - `filterbank`: the triangular filterbank and its normalization ([`MelNorm`])
//! - `options`: [`MelOptions`] and the log compression choice ([`LogSpec`])
//! - `spectrogram`: framing, batched FFT, mel projection, log compression
//!
//! Numerical parity against HuggingFace's `WhisperFeatureExtractor` lives in
//! `tests/whisper_mel_parity.rs`, which needs an out-of-repo fixture. The
//! inline tests pin the parts that can be checked in isolation: the two mel
//! scales, the filterbank shape, and the framing arithmetic.

mod filterbank;
mod options;
mod scale;
mod spectrogram;

pub use filterbank::MelNorm;
pub use options::{LogSpec, MelOptions};
pub use scale::{
    MelScale, hz_to_mel, hz_to_mel_slaney, mel_frequencies, mel_frequencies_with, mel_to_hz,
    mel_to_hz_slaney,
};
pub use spectrogram::{compute_mel_spectrogram, compute_mel_spectrogram_with};
