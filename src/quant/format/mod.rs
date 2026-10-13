//! Quantization format definitions (GGUF-compatible)
//!
//! Each format defines a fixed block structure. All invariants are encoded in the enum.
//!
//! # Packing axis contract (CRITICAL for dequant/quant_matmul agreement)
//!
//! Quantization is applied along the LAST axis (innermost, contiguous in memory).
//! For a weight matrix `[out_features, in_features]`, blocks run along `in_features`.
//! Example: `[4096, 4096]` with Q4_K (block_size=256) → each row = 16 blocks.

mod ids;
mod quant_format;

pub use quant_format::QuantFormat;
