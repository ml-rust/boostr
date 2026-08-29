//! Quantized tensor — block-structured storage for compressed model weights
//!
//! `QuantTensor<R>` is a separate type from `Tensor<R>`, NOT a custom DType.
//! Quantized data has block structure (not element structure) and supports
//! only three operations: storage, dequantization, and quantized matmul.

use crate::error::{Error, Result};
use crate::quant::{QuantFormat, QuantScheme};
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
/// - `storage` contains exactly `scheme.payload_bytes(shape)` bytes
/// - The last dimension of `shape` is a multiple of the scheme's unit width —
///   a GGUF block, or a TCF execution tile
/// - Under a GGUF scheme, blocks are packed along the last axis (contiguous in
///   memory). A TCF scheme is plane-major over the whole tensor instead, which
///   is why [`QuantScheme::is_row_blocked`] exists and why `gather_rows`
///   refuses a payload that is not row-blocked.
pub struct QuantTensor<R: Runtime> {
    /// Raw packed data on device
    storage: Storage<R>,
    /// Codec and byte addressing (determines the payload layout)
    scheme: QuantScheme,
    /// Logical shape in elements (not blocks)
    shape: Vec<usize>,
    /// Device where data lives
    device: R::Device,
}

/// Hand-written because `Storage<R>` and `R::Device` carry no `Debug` bound.
/// Prints the codec and the logical shape, which is what identifies a weight
/// in an error message; the payload bytes are never useful there.
impl<R: Runtime> core::fmt::Debug for QuantTensor<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuantTensor")
            .field("scheme", &self.scheme)
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
    /// device storage — can never drift apart on what counts as valid. The
    /// rules themselves live on [`QuantScheme`], which owns the size math for
    /// every codec.
    fn validate_shape_and_bytes(
        shape: &[usize],
        scheme: QuantScheme,
        storage_bytes: usize,
    ) -> Result<()> {
        scheme.validate(shape, storage_bytes)
    }

    /// Create a quantized tensor from raw packed data
    ///
    /// `data` must contain exactly `scheme.payload_bytes(shape)` bytes in the
    /// given scheme's own layout. A [`QuantFormat`] converts into a
    /// [`QuantScheme`], so a GGUF caller passes its format unchanged.
    ///
    /// # Errors
    ///
    /// - If the last dimension of `shape` is not a whole number of the
    ///   scheme's units
    /// - If `data` length doesn't match expected storage bytes
    pub fn from_bytes(
        data: &[u8],
        scheme: impl Into<QuantScheme>,
        shape: &[usize],
        device: &R::Device,
    ) -> Result<Self> {
        let scheme = scheme.into();
        Self::validate_shape_and_bytes(shape, scheme, data.len())?;

        // Store as U8 — the raw block bytes
        let storage =
            Storage::<R>::from_bytes(data, numr::dtype::DType::U8, device).map_err(Error::Numr)?;

        Ok(Self {
            storage,
            scheme,
            shape: shape.to_vec(),
            device: device.clone(),
        })
    }

    /// Create a quantized tensor from device storage that already holds
    /// tightly-packed block bytes — e.g. the output of [`Self::gather_rows`],
    /// where `index_select` produced a new U8 storage rather than a byte
    /// buffer on the host.
    ///
    /// Applies the same invariant checks as [`Self::from_bytes`] (see
    /// [`Self::validate_shape_and_bytes`]) against `storage`'s byte length,
    /// so a caller cannot alias in storage that doesn't actually match
    /// `format`/`shape`.
    ///
    /// # Errors
    ///
    /// - If the last dimension of `shape` is not a whole number of the
    ///   scheme's units
    /// - If `storage`'s byte length doesn't match the expected storage bytes
    pub fn from_storage(
        storage: Storage<R>,
        scheme: impl Into<QuantScheme>,
        shape: &[usize],
        device: &R::Device,
    ) -> Result<Self> {
        let scheme = scheme.into();
        Self::validate_shape_and_bytes(shape, scheme, storage.size_in_bytes())?;

        Ok(Self {
            storage,
            scheme,
            shape: shape.to_vec(),
            device: device.clone(),
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

        // A plane-major payload has no per-row byte run to gather: a row's
        // codes, scales, minima, and super values sit in four separate planes
        // spanning the whole tensor. Gathering its bytes would return a
        // plausible tensor of nonsense, so it is refused here.
        if !self.scheme.is_row_blocked() {
            return Err(Error::QuantError {
                reason: format!(
                    "gather_rows requires a row-blocked payload, and {} is plane-major",
                    self.scheme.name(),
                ),
            });
        }
        // `payload_bytes` IS the row-stride formula, with the divisibility
        // check built in — deriving it inline here would be a second copy of
        // the row stride, which is the class of duplication that has produced
        // silent corruption in this codebase before.
        let row_block_bytes = self.scheme.payload_bytes(&[cols])?;

        // `Storage` is Arc-shared, so this view is a cheap alias onto the
        // same device bytes, not a copy.
        let view =
            Tensor::<R>::from_storage_contiguous(self.storage.clone(), &[rows, row_block_bytes]);
        let gathered = client.index_select(&view, 0, indices)?;

        QuantTensor::from_storage(
            gathered.storage().clone(),
            self.scheme,
            &[indices.numel(), cols],
            &self.device,
        )
    }

    /// Codec and byte addressing of this tensor's payload.
    ///
    /// The general accessor: it answers for every codec, and a kernel that
    /// supports more than one dispatches on it.
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// The GGUF block format backing this tensor.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedQuantFormat`] when the payload follows another
    /// codec. A GGUF-only kernel calls this and propagates, which is how a
    /// TCF weight reaching a GGUF kernel becomes a named error instead of a
    /// misread block.
    pub fn format(&self) -> Result<QuantFormat> {
        self.scheme
            .gguf()
            .ok_or_else(|| Error::UnsupportedQuantFormat {
                format: format!("{} is not a GGUF block format", self.scheme.name()),
            })
    }

    /// Logical shape in elements
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of logical elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of independently decodable units: a GGUF block, or a TCF
    /// execution tile.
    pub fn num_blocks(&self) -> usize {
        self.scheme.decode_units(&self.shape)
    }

    /// Total storage size in bytes
    ///
    /// Equals `scheme.payload_bytes(shape)` by the constructor invariant, read
    /// off the storage so it holds for every codec without restating any
    /// codec's size math.
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
            scheme: self.scheme,
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

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

        assert_eq!(
            qt.format().expect("a GGUF tensor has a format"),
            QuantFormat::Q4_0
        );
        assert_eq!(qt.scheme(), QuantScheme::Gguf(QuantFormat::Q4_0));
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

    /// A TCF weight is carried by the SAME tensor type, at its packed size —
    /// no parallel tensor type, and no f32 blow-up.
    #[test]
    fn a_tcf_payload_is_carried_at_its_packed_size() {
        use crate::quant::TcfEncoding;
        use tcf_core::NativeEncoding;

        let device = cpu_device();
        let encoding = TcfEncoding::new(NativeEncoding::Q6S16DT64);
        // 15 tiles: three whole super-blocks and a partial fourth.
        let shape = [3usize, 320];
        let bytes = vec![0u8; 15 * 52 + 8];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, encoding, &shape, &device)
            .expect("a partial trailing super-block is legal");

        assert_eq!(qt.scheme(), QuantScheme::Tcf(encoding));
        assert_eq!(qt.shape(), &shape);
        assert_eq!(qt.numel(), 960);
        assert_eq!(qt.num_blocks(), 15);
        assert_eq!(qt.storage_bytes(), bytes.len());
        assert!(qt.format().is_err(), "a TCF tensor has no GGUF format");
    }

    /// Plane-major bytes have no per-row run, so a row gather is refused
    /// rather than returning bytes from the wrong planes.
    #[test]
    fn gather_rows_refuses_a_plane_major_payload() {
        use crate::quant::TcfEncoding;
        use tcf_core::NativeEncoding;

        let device = cpu_device();
        let client = CpuClient::new(device.clone());
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&[0u8; 2 * 36], encoding, &[2, 64], &device)
                .expect("two tiles");
        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).unwrap();

        let err = qt.gather_rows(&client, &indices).expect_err("refused");
        assert!(err.to_string().contains("plane-major"), "{err}");
    }

    #[test]
    fn test_gather_rows_rejects_non_2d() {
        let device = cpu_device();
        let client = CpuClient::new(device.clone());
        // Q4_0, 1-D shape [32] — gather_rows requires exactly 2 dims.
        let data = vec![0u8; 18];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[32], &device)
            .unwrap();
        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).unwrap();

        let result = qt.gather_rows(&client, &indices);
        assert!(result.is_err());
    }

    /// The whole safety net for silent row corruption: gathering rows from
    /// packed block bytes and dequantizing the result MUST agree, bit for
    /// bit, with dequantizing the whole table and then gathering rows of the
    /// dequantized floats. Any row-offset arithmetic bug in `gather_rows`
    /// would otherwise ship wrong embeddings without ever failing a test.
    #[test]
    fn test_gather_rows_matches_dequant_then_index_select_bit_for_bit() {
        use crate::quant::traits::{DequantOps, QuantizeOps};
        use numr::ops::IndexingOps;

        let device = cpu_device();
        let client = CpuClient::new(device.clone());

        // [8, 512]: 512 is a multiple of Q6_K's 256-element block size, so
        // each row is exactly 2 blocks.
        let rows = 8usize;
        let cols = 512usize;
        let source: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.037 - 3.0)
            .collect();
        let table = Tensor::<CpuRuntime>::from_slice(&source, &[rows, cols], &device).unwrap();

        let qt = client.quantize(&table, QuantFormat::Q6K).unwrap();

        // Repeated and out-of-order indices — exactly the pattern a real
        // token-ID batch produces.
        let idx_data = [3i64, 0, 7, 3];
        let indices =
            Tensor::<CpuRuntime>::from_slice(&idx_data, &[idx_data.len()], &device).unwrap();

        let gathered_quant = qt.gather_rows(&client, &indices).unwrap();
        let gathered_dequant = client
            .dequantize(&gathered_quant, numr::dtype::DType::F32)
            .unwrap();

        let whole_dequant = client.dequantize(&qt, numr::dtype::DType::F32).unwrap();
        let expected = client.index_select(&whole_dequant, 0, &indices).unwrap();

        let got_bits: Vec<u32> = gathered_dequant
            .to_vec::<f32>()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        let expected_bits: Vec<u32> = expected
            .to_vec::<f32>()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        assert_eq!(got_bits, expected_bits);
    }
}
