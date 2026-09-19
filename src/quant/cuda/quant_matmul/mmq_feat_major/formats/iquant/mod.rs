//! I-quant descriptors, grouped by staging mechanism.
//!
//! [`codebook`] indexes a flat 16-entry table. [`grid`] indexes a grid entry
//! through `mmqf_stage_iq_grid` at the IQ2 width; [`grid_iq3`] holds the
//! wider and affine grid formats. `IQ4_NL` sits in `legacy`, whose staged row
//! it shares.
//!
//! Every format resolves its index during staging. The staged row is one an
//! existing `vec_dot` already reads: Q8_0's at 32 elements per scale, Q6_K's
//! at 16, Q4_K's for IQ1_S, whose affine value needs a scale/min pair.

mod codebook;
mod grid;
mod grid_iq3;

pub(in crate::quant::cuda::quant_matmul) use codebook::IQ4_XS;
pub(in crate::quant::cuda::quant_matmul) use grid::{IQ2_S, IQ2_XS, IQ2_XXS};
pub(in crate::quant::cuda::quant_matmul) use grid_iq3::{IQ1_S, IQ3_S, IQ3_XXS};
