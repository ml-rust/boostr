//! Format-specific dispatch for CUDA quantized GEMV and tiled GEMM.
//!
//! - [`gemv`]           — GEMV path (M <= 64) for `quant_matmul`
//! - [`gemv_crossover`] — per-format largest `m` the GEMV path serves
//! - [`gemv_rows`]      — output columns per block for the prism `m = 1` kernel
//! - [`gemm`]           — tiled matmul path (M > 64) for `quant_matmul`

mod gemm;
mod gemv;
mod gemv_crossover;
mod gemv_rows;

pub(in crate::quant::cuda::quant_matmul) use gemm::{dispatch_matmul, feat_major_format};
pub(in crate::quant::cuda::quant_matmul) use gemv::dispatch_gemv;
pub(in crate::quant::cuda::quant_matmul) use gemv_crossover::gemv_max_m;
