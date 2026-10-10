//! Harmonic-plus-noise source-filter excitation for the ISTFTNet generator.
//!
//! Upstream `decoder.generator.m_source` = `SourceModuleHnNSF`:
//!
//! ```text
//! m_source
//! ├── l_sin_gen: SineGen(sampling_rate, harmonic_num=8)   # no parameters
//! ├── l_linear:  Linear(harmonic_num + 1 = 9, 1)           # fc.weight, fc.bias
//! └── l_tanh:    nn.Tanh                                    # no parameters
//! ```
//!
//! Flow per f0 sample:
//!
//! 1. For each harmonic `h ∈ {1, 2, …, harmonic_num+1}`, generate a sine wave
//!    at `h · f0 / sample_rate` cycles per sample, as upstream's `_f02sine`
//!    does: the per-sample increments are linearly downsampled by the
//!    upsample scale, summed at that rate, scaled back up, and linearly
//!    interpolated to the sample rate.
//! 2. Stack harmonics along a new last axis → `[B, T, harmonic_num+1]`.
//! 3. Voiced/unvoiced mask `uv = (f0 > threshold)`; voiced frames carry the
//!    harmonics plus a small amount of Gaussian noise, while unvoiced frames
//!    are driven entirely by Gaussian noise (`sine_amp / 3`). Each harmonic
//!    also receives a random initial phase.
//! 4. `l_linear` + `tanh` collapse the `harmonic_num+1` channels to a single
//!    time-domain excitation signal.
//!
//! The random initial phase and additive noise match the reference NSF
//! implementation. A seed makes both draws reproducible on one backend. Without
//! one, they come from the runtime's unseeded RNG.

mod sine_gen;
mod source_module;

pub use sine_gen::SineGen;
pub use source_module::SourceModuleHnNSF;
