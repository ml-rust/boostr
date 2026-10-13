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
//! `source_module` documents the flow per f0 sample.

mod sine_gen;
mod source_module;

pub use sine_gen::SineGen;
pub use source_module::SourceModuleHnNSF;
