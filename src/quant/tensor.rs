//! Quantized tensor — block-structured storage for compressed model weights
//!
//! `QuantTensor<R>` is a separate type from `Tensor<R>`, NOT a custom DType.
//! Quantized data has block structure (not element structure) and supports
//! only three operations: storage, dequantization, and quantized matmul.

use crate::error::{Error, Result};
use crate::quant::QuantFormat;
use numr::runtime::Runtime;
use numr::tensor::{Storage, Tensor};

/// Quantized tensor with block-structured storage
///
/// Unlike `Tensor<R>` which stores elements, `QuantTensor<R>` stores tightly-packed
/// blocks in a format-specific layout. The `shape` field stores the LOGICAL element
/// shape (not the block shape).
///
/// # Invariants
///
/// - `storage` contains exactly `format.storage_bytes(numel)` bytes
/// - The last dimension of `shape` is a multiple of the format's block size
/// - Blocks are packed along the last axis, contiguous in memory, so a row
///   is a contiguous byte run and `gather_rows` is a byte gather.
pub struct QuantTensor<R: Runtime> {
    /// Raw packed data on device
    storage: Storage<R>,
    /// Block format (determines the payload layout)
    format: QuantFormat,
    /// Logical shape in elements (not blocks)
    shape: Vec<usize>,
    /// Device where data lives
    device: R::Device,
    /// The activation contract the source format declared for this weight,
    /// when the format can declare one.
    ///
    /// `Some(..)` for a TCF weight: Section 3 requires every tensor to name a
    /// `ContractRecord`, and dispatch resolves against it. `None` for a GGUF
    /// weight, whose block formats have no field to carry one — absence means
    /// the format cannot say, never that any kernel will do. Set through
    /// [`QuantTensor::with_activation_contract`] and read through
    /// [`QuantTensor::activation_contract`], both in `crate::quant::contract`.
    pub(crate) contract: Option<crate::quant::contract::ActivationContract>,
}

/// Hand-written because `Storage<R>` and `R::Device` carry no `Debug` bound.
/// Prints the codec and the logical shape, which is what identifies a weight
/// in an error message; the payload bytes are never useful there.
impl<R: Runtime> core::fmt::Debug for QuantTensor<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuantTensor")
            .field("format", &self.format)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime<DType = numr::dtype::DType>> QuantTensor<R> {
    /// Check the invariants documented on [`QuantTensor`] against a candidate
    /// `shape` and the byte length of the storage that will back it.
    ///
    /// Shared by [`Self::from_bytes`] and [`Self::from_storage`] so the two
    /// constructors — one copying fresh bytes in, one aliasing existing
    /// device storage — can never drift apart on what counts as valid.
    fn validate_shape_and_bytes(
        shape: &[usize],
        format: QuantFormat,
        storage_bytes: usize,
    ) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::QuantError {
                reason: "QuantTensor shape must be non-empty".into(),
            });
        }
        let last_dim = shape.last().copied().unwrap_or(0);
        let unit = format.block_size();
        if unit == 0 || !last_dim.is_multiple_of(unit) {
            return Err(Error::QuantError {
                reason: format!(
                    "last dimension {last_dim} is not a multiple of {}'s block size {unit}",
                    format.name(),
                ),
            });
        }
        let expected = format.storage_bytes(shape.iter().product())?;
        if storage_bytes != expected {
            return Err(Error::QuantError {
                reason: format!(
                    "expected {expected} bytes for {} with shape {shape:?}, got {storage_bytes} bytes",
                    format.name(),
                ),
            });
        }
        Ok(())
    }

    /// Create a quantized tensor from raw packed data
    ///
    /// `data` must contain exactly `format.storage_bytes(numel)` bytes in the
    /// format's own block layout.
    ///
    /// # Errors
    ///
    /// - If the last dimension of `shape` is not a whole number of blocks
    /// - If `data` length doesn't match expected storage bytes
    pub fn from_bytes(
        data: &[u8],
        format: QuantFormat,
        shape: &[usize],
        device: &R::Device,
    ) -> Result<Self> {
        Self::validate_shape_and_bytes(shape, format, data.len())?;

        // Store as U8 — the raw block bytes
        let storage =
            Storage::<R>::from_bytes(data, numr::dtype::DType::U8, device).map_err(Error::Numr)?;

        Ok(Self {
            storage,
            format,
            shape: shape.to_vec(),
            device: device.clone(),
            contract: None,
        })
    }

    /// Create a quantized tensor from device storage that already holds
    /// tightly-packed block bytes — e.g. the output of [`Self::gather_rows`],
    /// where `index_select` produced a new U8 storage rather than a byte
    /// buffer on the host.
    ///
    /// Applies the same invariant checks as [`Self::from_bytes`] (see
    /// `Self::validate_shape_and_bytes`) against `storage`'s byte length,
    /// so a caller cannot alias in storage that doesn't actually match
    /// `format`/`shape`.
    ///
    /// # Errors
    ///
    /// - If the last dimension of `shape` is not a whole number of blocks
    /// - If `storage`'s byte length doesn't match the expected storage bytes
    pub fn from_storage(
        storage: Storage<R>,
        format: QuantFormat,
        shape: &[usize],
        device: &R::Device,
    ) -> Result<Self> {
        Self::validate_shape_and_bytes(shape, format, storage.size_in_bytes())?;

        Ok(Self {
            storage,
            format,
            shape: shape.to_vec(),
            device: device.clone(),
            contract: None,
        })
    }

    /// Gather rows from a `[rows, cols]` quantized weight table without
    /// dequantizing anything but the gathered rows.
    ///
    /// The block layout is packed along the last (column) axis, so a whole
    /// row is `cols / format.block_size()` contiguous blocks — exactly the
    /// unit `index_select` needs. This aliases the packed bytes as a `U8`
    /// tensor shaped `[rows, row_block_bytes]`, reuses numr's existing
    /// `index_select` to gather rows of that byte matrix (bounds-checked
    /// there already), then rewraps the gathered bytes as a `QuantTensor`.
    /// No dequantization happens here and no new kernel is needed: gathering
    /// whole rows of tightly-packed blocks is byte-for-byte the same
    /// operation as gathering rows of any other row-major matrix.
    ///
    /// # Errors
    ///
    /// - If this tensor's shape is not exactly 2-D
    pub fn gather_rows<C>(&self, client: &C, indices: &Tensor<R>) -> Result<QuantTensor<R>>
    where
        C: numr::ops::IndexingOps<R>,
    {
        if self.shape.len() != 2 {
            return Err(Error::QuantError {
                reason: format!(
                    "gather_rows requires a 2-D QuantTensor (rows, cols), got shape {:?}",
                    self.shape
                ),
            });
        }
        let rows = self.shape.first().copied().unwrap_or(0);
        let cols = self.shape.get(1).copied().unwrap_or(0);

        // `storage_bytes` IS the row-stride formula, with the divisibility
        // check built in — deriving it inline here would be a second copy of
        // the row stride, which is the class of duplication that has produced
        // silent corruption in this codebase before.
        let row_block_bytes = self.format.storage_bytes(cols)?;

        // `Storage` is Arc-shared, so this view is a cheap alias onto the
        // same device bytes, not a copy.
        let view =
            Tensor::<R>::from_storage_contiguous(self.storage.clone(), &[rows, row_block_bytes]);
        let gathered = client.index_select(&view, 0, indices)?;

        QuantTensor::from_storage(
            gathered.storage().clone(),
            self.format,
            &[indices.numel(), cols],
            &self.device,
        )
    }

    /// The block format backing this tensor.
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

    /// Number of blocks in the payload.
    pub fn num_blocks(&self) -> usize {
        self.numel()
            .checked_div(self.format.block_size())
            .unwrap_or(0)
    }

    /// Total storage size in bytes
    ///
    /// Equals `format.storage_bytes(numel)` by the constructor invariant, read
    /// off the storage so it holds without restating the size math.
    pub fn storage_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    /// Device where data lives
    pub fn device(&self) -> &R::Device {
        &self.device
    }

    /// Copy the tightly-packed block bytes to host memory
    ///
    /// Returns exactly [`Self::storage_bytes`] bytes, in the format's own block
    /// layout — the bytes a GGUF writer puts on disk verbatim.
    ///
    /// This exists because the writer side of the format needs the bytes and
    /// [`Self::storage`] cannot give them safely: `as_host_slice` is `unsafe`
    /// and only valid when the storage is host memory, so a consumer using it
    /// would silently read a device pointer on CUDA. This copies through the
    /// runtime instead and works on every backend.
    ///
    /// # Errors
    ///
    /// If the device-to-host copy fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.storage.try_to_vec::<u8>().map_err(Error::Numr)
    }

    /// Raw storage
    pub fn storage(&self) -> &Storage<R> {
        &self.storage
    }
}

// Manual, not derived: `#[derive(Clone)]` would add an `R: Clone` bound on
// the runtime type parameter itself, which no caller needs and no `Runtime`
// impl provides. `storage` is `Arc`-backed (see `Storage`), so this clone
// shares the underlying block bytes with the original rather than copying
// them — cheap, and the two `QuantTensor`s alias the same device memory.
impl<R: Runtime> Clone for QuantTensor<R> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            format: self.format,
            shape: self.shape.clone(),
            device: self.device.clone(),
            contract: self.contract.clone(),
        }
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
    fn test_to_bytes_round_trips_from_bytes() {
        let device = cpu_device();
        let data: Vec<u8> = (0..18u8).collect();
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[32], &device)
            .unwrap();

        let out = qt.to_bytes().unwrap();
        assert_eq!(out.len(), qt.storage_bytes());
        assert_eq!(out, data);
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
