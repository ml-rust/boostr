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
pub mod model;
pub mod nn;
pub mod ops;
pub mod optimizer;
pub mod quant;
pub mod trainer;

// Re-export primary boostr traits
pub use nn::{Init, VarBuilder, VarMap, Weight};
pub use ops::{
    AttentionOps, FlashAttentionOps, FusedFp8TrainingOps, FusedOptimizerOps, FusedQkvOps,
    KvCacheOps, MlaOps, PagedAttentionOps, RoPEOps, var_flash_attention,
};
pub use quant::{DequantOps, FusedQuantOps, QuantFormat, QuantMatmulOps, QuantTensor};

// Re-export numr types that users will commonly need
pub use numr::dtype::DType;
pub use numr::error::{Error as NumrError, Result as NumrResult};
pub use numr::runtime::{Runtime, RuntimeClient};
pub use numr::tensor::Tensor;

// Re-export runtime types for convenience (blazr uses boostr::CpuRuntime, etc.)
pub use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
pub use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

// Re-export numr modules for path-based access (e.g., boostr::runtime::Device)
pub use numr::runtime;
pub use numr::tensor;

// Re-export TensorOps as a trait alias that blazr uses for client bounds
pub use ops::TensorOps;

// Re-export IndexingOps for KV cache bounds
pub use numr::ops::traits::IndexingOps;

// Re-export ScalarOps for blazr's temperature scaling
pub use numr::ops::ScalarOps;

// Re-export numr ops needed by blazr's Mamba2 inference path
pub use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, TypeConversionOps, UnaryOps,
};

#[cfg(test)]
pub(crate) mod test_utils {
    use numr::runtime::cpu::{CpuClient, CpuDevice};

    /// Create a CPU client and device for use in unit tests.
    pub(crate) fn cpu_setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }
}
