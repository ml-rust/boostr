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
//!   residual path. The STFT analysis uses [`crate::model::audio::stft`].
//!
//! Full path (`forward_cpu_full`):
//!
//! ```text
//! har = harmonic_excitation_spec(f0)            # [B, n_fft+2, T] via STFT
//! for each upsample stage i:
//!     x = leaky_relu(x)
//!     x_source = noise_res[i](noise_convs[i](har), style)
//!     x = ups[i](x) + x_source                  # cropped to trunk length
//!     x = mean(resblocks[i*K .. (i+1)*K](x, style))
//! x = leaky_relu(x)
//! (mag, phase) = conv_post(x)
//! ```
//!
//! # Known limitation
//!
//! **Reflection padding.** Upstream applies `ReflectionPad1d(3)` to the last
//! upsample stage before the residual add; this build uses the STFT
//! `center=True` framing plus right-cropping to align trunk and source lengths,
//! which matches output length but differs from reflection padding at the
//! boundary by a few samples. See [`IStftNetGenerator::forward_cpu_full`].

mod core;
mod cpu_full;
#[cfg(test)]
mod test_support;

pub use core::{GeneratorStftParams, IStftNetGenerator, IStftNetGeneratorOpts};
