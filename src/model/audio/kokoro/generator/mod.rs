//! `IStftNetGenerator` — Kokoro's spectrogram decoder.
//!
//! Upstream `decoder.generator` composes:
//!
//! ```text
//! m_source  : SourceModuleHnNSF                           (f0 → excitation)
//! f0_upsamp : nn.Upsample                                 (f0 rate-matching)
//! stft      : TorchSTFT(n_fft, hop, win)                  (noise-path spectral analysis)
//! ups[k]    : weight-normed ConvTranspose1d               (main-path upsampling)
//! noise_convs[k] : plain Conv1d                           (harmonic-spectrum conditioning)
//! noise_res[k]   : AdaINResBlock1                         (per-stage noise residuals)
//! resblocks[k * num_kernels + j] : AdaINResBlock1         (per-stage main residuals)
//! conv_post : weight-normed Conv1d → (exp|sin) split      (mag/phase head)
//! ```
//!
//! [`IStftNetGenerator::forward`] runs the complete vocoder on any runtime,
//! including the harmonic-excitation → STFT → per-stage noise residual path.
//! The STFT analysis uses [`crate::model::audio::stft()`]. A generator
//! loaded without noise weights runs the upsampling stages alone.
//! [`IStftNetGenerator::forward_with_spectrum`] runs everything after the
//! source over a given excitation spectrum, with no random draw. `full`
//! documents the full path.

mod core;
mod full;
#[cfg(test)]
mod test_support;

pub use core::{GeneratorStftParams, IStftNetGenerator, IStftNetGeneratorOpts};
