//! VoxCPM2's `fsq_layer`: a finite-scalar-quantization bottleneck between the
//! `base_lm` decoder and `feat_decoder`'s DiT, plus the `stop` classifier
//! chain that shares its input width.
//!
//! Reference: `ScalarQuantizationLayer.forward`, EVAL mode only. The
//! reference's training branch runs a straight-through estimator around
//! `torch.round`; boostr is inference-only, so that branch is dead code and
//! is deliberately NOT ported here.
//!
//! - `quantization`: [`ScalarQuantization`], the bottleneck itself
//! - `aux`: [`AuxProjections`], the six root-level projections around it

mod aux;
mod quantization;

pub use aux::AuxProjections;
pub use quantization::ScalarQuantization;
