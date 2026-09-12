//! Format-specific dispatch for CUDA quantized GEMV and tiled GEMM.
//!
//! - [`gemv`]   — GEMV path (M <= 64) for `quant_matmul`
//! - [`gemm`]   — tiled matmul path (M > 64) for `quant_matmul`

mod gemm;
mod gemv;

pub(in crate::quant::cuda::quant_matmul) use gemm::{dispatch_matmul, feat_major_format};
pub(in crate::quant::cuda::quant_matmul) use gemv::{dispatch_gemv, gemv_max_m};
