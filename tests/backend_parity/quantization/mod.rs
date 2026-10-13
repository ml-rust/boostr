//! Backend parity: calibration, quantized matmul, INT4 GEMM and NF4.

pub use super::helpers;

pub mod calibration;
pub mod quant_fused;
pub mod quant_int4_gemm;
pub mod quant_matmul_batch;
// Imports the CUDA-only forced-tile hook, which the CPU-only build lacks.
#[cfg(feature = "cuda")]
pub mod quant_matmul_feat_tile;
pub mod quant_matmul_gguf_gemm;
pub mod quant_matmul_q4k;
pub mod quant_matmul_q8_0;
pub mod quant_nf4;
