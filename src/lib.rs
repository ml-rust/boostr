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

pub mod error;
pub mod format;
pub mod inference;
pub mod model;
pub mod nn;
pub mod ops;
pub mod optimizer;
pub mod quant;

// Re-export primary boostr traits
pub use ops::{AttentionOps, RoPEOps};
pub use quant::{DequantOps, QuantFormat, QuantMatmulOps, QuantTensor};

// Re-export numr types that users will commonly need
pub use numr::dtype::DType;
pub use numr::error::{Error as NumrError, Result as NumrResult};
pub use numr::runtime::{Runtime, RuntimeClient};
pub use numr::tensor::Tensor;

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
