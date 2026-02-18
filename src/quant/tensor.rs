//! Quantized tensor — block-structured storage for compressed model weights
//!
//! `QuantTensor<R>` is a separate type from `Tensor<R>`, NOT a custom DType.
//! Quantized data has block structure (not element structure) and supports
//! only three operations: storage, dequantization, and quantized matmul.

use crate::error::{Error, Result};
use crate::quant::QuantFormat;
use numr::runtime::Runtime;
use numr::tensor::Storage;

/// Quantized tensor with block-structured storage
///
/// Unlike `Tensor<R>` which stores elements, `QuantTensor<R>` stores tightly-packed
/// blocks in a format-specific layout. The `shape` field stores the LOGICAL element
/// shape (not the block shape).
///
/// # Invariants
///
/// - `storage` contains exactly `format.storage_bytes(numel)` bytes
/// - The last dimension of `shape` is a multiple of `format.block_size()`
/// - Blocks are packed along the last axis (contiguous in memory)
pub struct QuantTensor<R: Runtime> {
    /// Raw block data on device
    storage: Storage<R>,
    /// Quantization format (determines block layout)
    format: QuantFormat,
    /// Logical shape in elements (not blocks)
    shape: Vec<usize>,
    /// Device where data lives
    device: R::Device,
}

impl<R: Runtime<DType = numr::dtype::DType>> QuantTensor<R> {
    /// Create a quantized tensor from raw block data
    ///
    /// `data` must contain exactly `format.storage_bytes(numel)` bytes of
    /// tightly-packed blocks in the given format.
    ///
    /// # Errors
    ///
    /// - If the last dimension of `shape` is not a multiple of `format.block_size()`
    /// - If `data` length doesn't match expected storage bytes
    pub fn from_bytes(
        data: &[u8],
        format: QuantFormat,
        shape: &[usize],
        device: &R::Device,
    ) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::QuantError {
                reason: "QuantTensor shape must be non-empty".into(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim % format.block_size() != 0 {
            return Err(Error::QuantError {
                reason: format!(
                    "last dimension {} is not a multiple of {}'s block_size {}",
                    last_dim,
                    format.name(),
                    format.block_size(),
                ),
            });
        }

        let numel: usize = shape.iter().product();
        let expected_bytes = format.storage_bytes(numel)?;
        if data.len() != expected_bytes {
            return Err(Error::QuantError {
                reason: format!(
                    "expected {} bytes for {} with {} elements, got {} bytes",
                    expected_bytes,
                    format.name(),
                    numel,
                    data.len(),
                ),
            });
        }

        // Store as U8 — the raw block bytes
        let storage =
            Storage::<R>::from_bytes(data, numr::dtype::DType::U8, device).map_err(Error::Numr)?;

        Ok(Self {
            storage,
            format,
            shape: shape.to_vec(),
            device: device.clone(),
        })
    }

    /// Quantization format
    pub fn format(&self) -> QuantFormat {
        self.format
    }

    /// Logical shape in elements
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of logical elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.numel() / self.format.block_size()
    }

    /// Total storage size in bytes
    pub fn storage_bytes(&self) -> usize {
        self.num_blocks() * self.format.block_bytes()
    }

    /// Device where data lives
    pub fn device(&self) -> &R::Device {
        &self.device
    }

    /// Raw storage
    pub fn storage(&self) -> &Storage<R> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn cpu_device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_create_q4_0() {
        // Q4_0: 32 elements per block, 18 bytes per block
        let device = cpu_device();
        let data = vec![0u8; 18]; // 1 block = 32 elements
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[32], &device)
            .unwrap();

        assert_eq!(qt.format(), QuantFormat::Q4_0);
        assert_eq!(qt.shape(), &[32]);
        assert_eq!(qt.numel(), 32);
        assert_eq!(qt.num_blocks(), 1);
        assert_eq!(qt.storage_bytes(), 18);
    }

    #[test]
    fn test_create_q4k_matrix() {
        // Q4K: 256 elements per block, 144 bytes per block
        // Shape [4096, 4096] → 16 blocks per row, 4096 rows
        let device = cpu_device();
        let numel = 4096 * 4096;
        let num_blocks = numel / 256;
        let data = vec![0u8; num_blocks * 144];
        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4K, &[4096, 4096], &device)
                .unwrap();

        assert_eq!(qt.shape(), &[4096, 4096]);
        assert_eq!(qt.numel(), numel);
        assert_eq!(qt.num_blocks(), num_blocks);
    }

    #[test]
    fn test_alignment_error() {
        let device = cpu_device();
        let data = vec![0u8; 18];
        // 33 is not a multiple of 32 (Q4_0 block_size)
        let result =
            QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[33], &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_mismatch_error() {
        let device = cpu_device();
        let data = vec![0u8; 10]; // Wrong size (should be 18 for Q4_0 × 32 elements)
        let result =
            QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[32], &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_shape_error() {
        let device = cpu_device();
        let data = vec![0u8; 18];
        let result = QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[], &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_block() {
        let device = cpu_device();
        // 4 blocks of Q8_0 (32 elements, 34 bytes each) = 128 elements
        let data = vec![0u8; 4 * 34];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q8_0, &[128], &device)
            .unwrap();

        assert_eq!(qt.num_blocks(), 4);
        assert_eq!(qt.storage_bytes(), 136);
    }
}
