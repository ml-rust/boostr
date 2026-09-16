//! CPU quantized matmul kernels
//!
//! Dequantize-and-accumulate per block row for cache efficiency.
//! Computes: activation [M, K] × weight^T → output [M, N]
//!
//! Weight is stored as [N, K] (N output rows, K input cols each), matching
//! the packing axis contract: quantization blocks run along the last (K) axis.
//! We iterate over weight rows (output columns), dequantize one row at a time,
//! and accumulate the dot product contribution.
//!
//! Optimizations:
//! - Rayon parallelism over N (weight rows / output columns)
//! - Thread-local dequant buffers to avoid contention
//! - AVX2+FMA SIMD dot product

mod batch;
mod shared;
mod single;

pub use batch::quant_matmul_batch_f32;
pub use shared::dequant_row_f32;
pub use single::quant_matmul_f32;
