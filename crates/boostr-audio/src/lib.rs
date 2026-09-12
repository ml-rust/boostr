//! Sample-level audio DSP, file codecs, voices, and product pipelines.
//!
//! Placement rule: code that operates on `Tensor<R>` with backend kernels
//! belongs in `boostr::model::audio`. Code that operates on `Vec<f32>`
//! samples, files, directories, voices, or text pipelines belongs here.
