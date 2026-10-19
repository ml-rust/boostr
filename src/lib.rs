//! # boostr
//!
//! **ML framework built on numr — attention, quantization, model architectures.**
//!
//! boostr extends numr's foundational numerical computing with ML-specific operations,
//! quantized tensor support, and model building blocks. It uses numr's runtime, tensors,
//! and ops directly — no reimplementation, no wrappers.
//!
//! ## Relationship to numr
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    boostr ◄── YOU ARE HERE               │
//! │   (attention, RoPE, MoE, quantization, model loaders)   │
//! └──────────────────────────┬──────────────────────────────┘
//! │                      numr                                │
//! │     (tensors, ops, runtime, autograd, linalg, FFT)       │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design
//!
//! - **Extension traits**: ML ops (AttentionOps, RoPEOps) implemented on numr's clients
//! - **QuantTensor**: Separate type for block-quantized data (GGUF formats)
//! - **impl_generic**: Composite ops composed from numr primitives, same on all backends
//! - **Custom kernels**: Dequant, quantized matmul, fused attention (SIMD/PTX/WGSL)

pub mod data;
pub mod distributed;
pub mod error;
pub mod format;
pub mod inference;
#[cfg(feature = "cuda")]
mod kernel_loader;
pub mod model;
pub mod nn;
pub mod ops;
pub mod optimizer;
pub mod quant;
pub mod readback;
pub mod tcf;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod trainer;

// Re-export primary boostr traits
pub use nn::{Init, VarBuilder, VarMap, Weight, router_z_loss};
pub use ops::{
    AttentionOps, AttnOutLayout, DeviceGrammarDfa, FilterRule, FlashAttentionOps,
    FusedFp8TrainingOps, FusedOptimizerOps, FusedQkvOps, GatedDeltaNetOps, GdnGrads, GrammarDfaOps,
    KvCacheOps, KvSeqLen, MlaOps, PagedAttentionOps, RoPEOps, RoPEPackedOps, SampleSeed,
    SamplingOps, SamplingProbsOps, SamplingScratch, var_flash_attention, var_gdn_chunk,
};
pub use quant::{
    DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor, DequantOps, FusedQuantOps,
    QuantFormat, QuantMatmulOps, QuantTensor, QuantizeOps, ScheduleTuning,
};
pub use readback::scalar_f32;

// Re-export numr types that users will commonly need
pub use numr::dtype::DType;
pub use numr::error::{Error as NumrError, Result as NumrResult};
pub use numr::runtime::{Runtime, RuntimeClient};
pub use numr::tensor::Tensor;

// Re-export runtime types so a caller can name `boostr::CpuRuntime` and friends
pub use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
pub use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

// Re-export numr modules for path-based access (e.g., boostr::runtime::Device)
pub use numr::autograd;
pub use numr::runtime;
pub use numr::tensor;

// Re-export TensorOps as a trait alias for a caller's client bounds
pub use ops::TensorOps;

// Re-export MoE expert weight types for a caller's expert offloading
pub use model::ExpertWeights;

// Re-export the embedding pipeline for sentence embedding use cases
pub use model::encoder::{EmbeddingPipeline, Encoder, EncoderClient, EncoderConfig, Pooling};

// Re-export IndexingOps for KV cache bounds
pub use numr::ops::traits::IndexingOps;

// Re-export ScalarOps for a caller's temperature scaling
pub use numr::ops::ScalarOps;

// Re-export HostCopyOps so a caller can write a pre-allocated device buffer in
// place — the way the lane table and the decode graph's own buffers are
// written — without a direct numr dependency.
pub use numr::ops::HostCopyOps;

// Re-export numr ops a caller's Mamba2 inference path needs
pub use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, TypeConversionOps, UnaryOps,
};

// Re-export the remaining numr op traits that appear in boostr's own public
// signatures (MoE routing, attention, SSM). Anyone building a trainer or
// inference server on boostr has to name these to satisfy the `C:` bounds on
// `forward`, so they belong on boostr's surface rather than forcing a direct
// numr dependency just to spell a bound.
pub use numr::ops::{
    CompareOps, ConditionalOps, LinalgOps, MatmulOps, RandomOps, ReduceOps, ShapeOps, SortingOps,
};

// PaddingMode is part of Conv1d's public signature, so downstream crates need
// it to construct a conv layer without depending on numr directly.
pub use numr::ops::PaddingMode;

/// Pre-load all CUDA PTX modules needed for LLaMA inference.
///
/// This front-loads all PTX→SASS JIT compilation during warmup,
/// eliminating the first-decode-token JIT latency spike.
/// Call this once after creating the CudaClient, before any real inference.
#[cfg(feature = "cuda")]
pub fn preload_inference_modules(client: &CudaClient) -> Result<(), error::Error> {
    use numr::runtime::Device;
    use numr::runtime::cuda::kernels::kernel_names;

    // numr core modules used by LLaMA inference
    client
        .preload_modules(&[
            kernel_names::BINARY_MODULE,
            kernel_names::UNARY_MODULE,
            kernel_names::SCALAR_MODULE,
            kernel_names::REDUCE_MODULE,
            kernel_names::ACTIVATION_MODULE,
            kernel_names::SOFTMAX_MODULE,
            kernel_names::NORM_RMS_MODULE,
            kernel_names::NORM_LAYER_MODULE,
            kernel_names::FUSED_ADD_RMS_NORM_MODULE,
            kernel_names::FUSED_ADD_LAYER_NORM_MODULE,
            kernel_names::CAST_MODULE,
            kernel_names::UTILITY_MODULE,
            kernel_names::MATMUL_MODULE,
            kernel_names::GEMV_MODULE,
        ])
        .map_err(error::Error::Numr)?;

    // boostr ops modules
    ops::cuda::kernels::preload_modules(
        client.context(),
        client.device().id(),
        &[
            ops::cuda::kernels::ROPE_MODULE,
            ops::cuda::kernels::DECODE_ATTENTION_MODULE,
            ops::cuda::kernels::PAGED_DECODE_ATTENTION_MODULE,
            ops::cuda::kernels::PAGED_ATTENTION_MODULE,
            ops::cuda::kernels::FLASH_V2_MODULE,
            ops::cuda::kernels::KV_CACHE_UPDATE_MODULE,
            ops::cuda::kernels::RESHAPE_AND_CACHE_MODULE,
        ],
    )?;

    // boostr quant modules (for GGUF inference)
    quant::cuda::kernels::preload_modules(
        client.context(),
        client.device().id(),
        &[
            quant::cuda::kernels::DEQUANT_MODULE,
            quant::cuda::kernels::QUANT_MATMUL_MODULE,
            quant::cuda::kernels::QUANT_GEMV_MODULE,
            quant::cuda::kernels::QUANT_ACT_MODULE,
        ],
    )?;

    Ok(())
}
