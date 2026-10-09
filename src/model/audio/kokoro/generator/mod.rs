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
//! Two forward paths are provided:
//!
//! * [`IStftNetGenerator::forward`] — the generic, runtime-agnostic main path
//!   (no source/noise conditioning). Used by generic-runtime callers and when
//!   a checkpoint ships without noise weights.
//! * [`IStftNetGenerator::forward_cpu_full`] — the complete CPU vocoder used in
//!   production, including the harmonic-excitation → STFT → per-stage noise
//!   residual path. The STFT analysis uses [`crate::model::audio::stft()`].
//!
//! Full path (`forward_cpu_full`):
//!
//! ```text
//! har = harmonic_excitation_spec(f0)            # [B, n_fft+2, T] via STFT
//! for each upsample stage i:
//!     x = leaky_relu(x)
//!     x_source = noise_res[i](noise_convs[i](har), style)
//!     x = ups[i](x)                             # last stage: reflect-pad 1 left
//!     x = x + x_source
//!     x = mean(resblocks[i*K .. (i+1)*K](x, style))
//! x = leaky_relu(x, 0.01)
//! (mag, phase) = conv_post(x)
//! ```
//!
//! The last stage pads the trunk with one reflected frame on the left
//! before the excitation add, as upstream's `ReflectionPad1d((1, 0))` does,
//! so the trunk meets the excitation's `center=True` frame count.

mod core;
mod cpu_full;
#[cfg(test)]
mod test_support;

pub use core::{GeneratorStftParams, IStftNetGenerator, IStftNetGeneratorOpts};
