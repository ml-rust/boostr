//! [`WeightSource`]: one named-tensor read contract over every checkpoint
//! format, plus the two decorators that change WHAT it yields rather than
//! where the bytes come from.
//!
//! Placed under `format` because it is a file-format concern and nothing
//! else: safetensors, GGUF and TCF each implement it, and no model
//! architecture appears anywhere in it. It lived under
//! `model::audio::voxcpm::loader::support` while VoxCPM2 was its only
//! caller; `model::audio` is behind the `audio` feature, so a text-model
//! loader could not reach it there.

pub mod dense;
pub mod source;
pub mod tcf;

pub use dense::DenseWeightSource;
pub use source::WeightSource;
pub use tcf::TcfSource;
