//! GGUF file reader
//!
//! Parses the GGUF binary format (v1-v3) and provides access to metadata
//! and tensor data. F32 tensors are loaded as `Tensor<R>`, quantized tensors
//! as `QuantTensor<R>`.

use super::GgufTensorInfo;
use super::io::{
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, align_offset, read_kv_pair, read_tensor_info, read_u32,
    read_u64,
};
use super::metadata::GgufMetadata;
use super::types::GgmlType;
use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use memmap2::Mmap;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Tensors whose dequantized f32 output exceeds this byte threshold use the
/// streaming dequant path, which processes one chunk at a time and uploads
/// each chunk directly into the pre-allocated device buffer.  Tensors below
/// this threshold use the original one-shot allocation path.
///
/// 64 MiB keeps the transient CPU buffer well within L3 / RSS limits even on
/// memory-constrained systems, while keeping the number of upload calls low
/// (a 1 GiB embedding table becomes ~16 chunks).
const STREAMING_THRESHOLD: usize = 64 * 1024 * 1024; // 64 MiB in bytes

/// Backing storage for GGUF tensor data.
enum GgufStorage {
    File(File),
    Mmap(Mmap),
    /// In-memory buffer (for wasm or data loaded via HTTP).
    InMemory(Vec<u8>),
}

/// GGUF file reader
pub struct Gguf {
    storage: GgufStorage,
    version: u32,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensorInfo>,
    data_offset: u64,
}

impl Gguf {
    /// Open and parse a GGUF file using regular file I/O.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_impl(path, false)
    }

    /// Open a GGUF file with optional memory-mapping.
    ///
    /// When `use_mmap` is `true`, the tensor data region is memory-mapped for
    /// zero-copy reads. The header and metadata are still read via buffered I/O.
    /// Falls back to regular file I/O if mmap fails (e.g. on unsupported platforms).
    pub fn open_with_mmap<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        Self::open_impl(path, use_mmap)
    }

    /// Parse a GGUF model from an in-memory byte buffer.
    ///
    /// This is the primary entry point for wasm/browser usage where models are
    /// fetched via HTTP and provided as byte slices. The entire buffer is kept
    /// in memory — tensor reads are zero-copy slices into this buffer.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(&data);
        let (version, metadata, tensors, data_offset) = Self::parse_header(&mut cursor)?;

        Ok(Gguf {
            storage: GgufStorage::InMemory(data),
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Parse GGUF header (magic, version, metadata, tensor info) from any `Read + Seek`.
    fn parse_header<R: Read + Seek>(
        reader: &mut R,
    ) -> Result<(u32, GgufMetadata, HashMap<String, GgufTensorInfo>, u64)> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(Error::ModelError {
                reason: format!("invalid GGUF magic: 0x{magic:08x}"),
            });
        }

        let version = read_u32(reader)?;
        if !(1..=3).contains(&version) {
            return Err(Error::ModelError {
                reason: format!("unsupported GGUF version: {version}"),
            });
        }

        let tensor_count = read_u64(reader)?;
        let kv_count = read_u64(reader)?;

        let mut metadata = GgufMetadata::default();
        for _ in 0..kv_count {
            let (key, value) = read_kv_pair(reader, version)?;
            metadata.kv.insert(key, value);
        }

        let alignment = metadata
            .get_u32("general.alignment")
            .map(|v| v as usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = read_tensor_info(reader, version)?;
            tensors.insert(info.name.clone(), info);
        }

        let current_pos = reader.stream_position().map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let data_offset = align_offset(current_pos, alignment);

        Ok((version, metadata, tensors, data_offset))
    }

    fn open_impl<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        let mut file = File::open(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let mut reader = BufReader::new(&mut file);

        let (version, metadata, tensors, data_offset) = Self::parse_header(&mut reader)?;

        // Drop the BufReader so we can take ownership of `file` again.
        drop(reader);

        let storage = if use_mmap {
            // SAFETY: The file is read-only and we do not mutate the mapping.
            // The caller must not truncate or replace the file while the Gguf is live.
            match unsafe { Mmap::map(&file) } {
                Ok(mmap) => GgufStorage::Mmap(mmap),
                Err(_) => GgufStorage::File(file), // graceful fallback
            }
        } else {
            GgufStorage::File(file)
        };

        Ok(Gguf {
            storage,
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::ModelError {
            reason: format!("GGUF tensor not found: {name}"),
        })
    }

    /// Read a byte slice from storage at the given absolute offset.
    fn read_slice(&mut self, abs_offset: u64, size: usize, name: &str) -> Result<Vec<u8>> {
        match &mut self.storage {
            GgufStorage::Mmap(mmap) => {
                let start = abs_offset as usize;
                let end = start + size;
                if end > mmap.len() {
                    return Err(Error::ModelError {
                        reason: format!(
                            "tensor '{name}' data at [{start}..{end}) exceeds mmap length {}",
                            mmap.len()
                        ),
                    });
                }
                Ok(mmap[start..end].to_vec())
            }
            GgufStorage::InMemory(buf) => {
                let start = abs_offset as usize;
                let end = start + size;
                if end > buf.len() {
                    return Err(Error::ModelError {
                        reason: format!(
                            "tensor '{name}' data at [{start}..{end}) exceeds buffer length {}",
                            buf.len()
                        ),
                    });
                }
                Ok(buf[start..end].to_vec())
            }
            GgufStorage::File(file) => {
                let mut buf = vec![0u8; size];
                file.seek(SeekFrom::Start(abs_offset))
                    .map_err(|e| Error::ModelError {
                        reason: format!("IO seek error: {e}"),
                    })?;
                file.read_exact(&mut buf).map_err(|e| Error::ModelError {
                    reason: format!("IO read error: {e}"),
                })?;
                Ok(buf)
            }
        }
    }

    /// Read raw tensor data bytes.
    ///
    /// When backed by mmap or in-memory buffer, this copies from the backing store.
    /// When backed by a file, this seeks and reads.
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        let abs_offset = self.data_offset + info.offset;
        let size = info.size_bytes();

        self.read_slice(abs_offset, size, name)
    }

    /// Load an F32 tensor (for unquantized F32/F16/BF16 tensors, converted to F32)
    pub fn load_tensor_f32<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        let bytes = self.read_tensor_bytes(name)?;

        // GGUF stores shape in GGML order (innermost first), reverse for row-major
        let mut shape = info.shape.clone();
        shape.reverse();

        let data: Vec<f32> = match info.ggml_type {
            GgmlType::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlType::F16 => bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            GgmlType::BF16 => bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect(),
            GgmlType::F64 => bytes
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            other => {
                // Dequantize quantized types to f32
                let format = other.to_quant_format().ok_or_else(|| Error::ModelError {
                    reason: format!(
                        "tensor '{name}' has type {other:?} which cannot be dequantized"
                    ),
                })?;
                let numel: usize = shape.iter().product();
                let block_size = format.block_size();
                let block_bytes = format.block_bytes();
                let row_k = info.shape[0]; // innermost dim (before reversal) = K per row
                let row_bytes = row_k / block_size * block_bytes;
                let n_rows = numel / row_k;
                let expected_bytes = n_rows * row_bytes;
                if bytes.len() < expected_bytes {
                    return Err(Error::ModelError {
                        reason: format!(
                            "tensor '{name}': expected at least {expected_bytes} bytes for dequantization, got {}",
                            bytes.len()
                        ),
                    });
                }
                let mut data = vec![0.0f32; numel];
                for row in 0..n_rows {
                    let src = &bytes[row * row_bytes..(row + 1) * row_bytes];
                    let dst = &mut data[row * row_k..(row + 1) * row_k];
                    crate::quant::cpu::kernels::quant_matmul::dequant_row_f32(src, dst, format);
                }
                data
            }
        };

        Tensor::<R>::try_from_slice(&data, &shape, device).map_err(Error::Numr)
    }

    /// Load an F32 tensor using a streaming dequant path for large tensors.
    ///
    /// For tensors whose dequantized size is ≤ `STREAMING_THRESHOLD` bytes this
    /// is identical to [`load_tensor_f32`].  For larger tensors the method:
    ///
    /// 1. Allocates the destination `Tensor<R>` of the correct shape once on the
    ///    device via `Tensor::try_empty`.
    /// 2. Dequantizes `STREAMING_CHUNK_ELEMS` f32 elements at a time on the CPU.
    /// 3. Uploads each chunk directly into the pre-allocated device buffer at the
    ///    correct byte offset, then drops the chunk `Vec<f32>` before the next
    ///    iteration.
    ///
    /// This bounds the transient CPU heap usage to `STREAMING_THRESHOLD` bytes
    /// instead of `numel × 4` bytes, preventing heap OOM for large quantized
    /// weights such as the token-embedding table in bge-reranker-v2-m3.
    ///
    /// The GPU-side allocation is a single contiguous buffer of `numel × 4`
    /// bytes; if the device cannot satisfy that allocation `Tensor::try_empty`
    /// returns `OutOfMemory` immediately (no change from the non-streaming path).
    ///
    /// Non-quantized types (F32 / F16 / BF16 / F64) always use the one-shot
    /// path regardless of size because their source bytes are already in the
    /// correct format and a single `copy_to_device` is optimal.
    pub fn load_tensor_f32_streaming<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        self.load_tensor_f32_streaming_impl::<R>(name, device, STREAMING_THRESHOLD)
    }

    /// Internal implementation shared by the public API and tests.
    ///
    /// `threshold_bytes` controls when to use the streaming path. Pass `0` in
    /// tests to force streaming even for small tensors.
    pub(super) fn load_tensor_f32_streaming_impl<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
        threshold_bytes: usize,
    ) -> Result<Tensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        // For non-quantized types there is no block structure to chunk over;
        // the bytes map 1-to-1 (or 2-to-1) to the output f32 values.  Use the
        // existing one-shot path which is already optimal.
        let is_quantized = matches!(
            info.ggml_type,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
        );

        let mut shape = info.shape.clone();
        shape.reverse();
        let numel: usize = shape.iter().product();

        // Fall back to the one-shot path for small tensors or non-quantized types.
        if !is_quantized || numel * 4 <= threshold_bytes {
            return self.load_tensor_f32(name, device);
        }

        // --- Streaming dequant path ---

        // All quantized formats use a 1-D row structure where shape[0] (before
        // reversal) is the innermost / fastest-varying dimension K.
        let format = info
            .ggml_type
            .to_quant_format()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "tensor '{name}' type {:?} cannot be dequantized",
                    info.ggml_type
                ),
            })?;

        let block_size = format.block_size();
        let block_bytes = format.block_bytes();
        let row_k = info.shape[0]; // innermost dim (GGML order, before reversal)
        let row_bytes = row_k / block_size * block_bytes;
        let n_rows = numel / row_k;
        let expected_bytes = n_rows * row_bytes;

        // Read all quantized source bytes once (they are compressed; ~¼ the
        // size of the f32 output).
        let abs_offset = self.data_offset + info.offset;
        let raw = self.read_slice(abs_offset, expected_bytes, name)?;

        // Allocate the destination tensor on the device.
        let dst = Tensor::<R>::try_empty(&shape, DType::F32, device).map_err(Error::Numr)?;
        let base_ptr = dst.ptr();

        // How many rows fit in one chunk?  Use the chunk size that corresponds
        // to `threshold_bytes`; when threshold_bytes == 0 (test mode) use 1 row.
        let chunk_elems_limit = if threshold_bytes == 0 {
            row_k
        } else {
            threshold_bytes / 4
        };
        let chunk_rows = (chunk_elems_limit / row_k).max(1);

        let mut row_start = 0usize;
        while row_start < n_rows {
            let row_end = (row_start + chunk_rows).min(n_rows);
            let chunk_elems = (row_end - row_start) * row_k;

            // Dequantize chunk into a temporary CPU buffer.
            let mut chunk_f32 = vec![0.0f32; chunk_elems];
            for (local_row, abs_row) in (row_start..row_end).enumerate() {
                let src = &raw[abs_row * row_bytes..(abs_row + 1) * row_bytes];
                let dst_slice = &mut chunk_f32[local_row * row_k..(local_row + 1) * row_k];
                crate::quant::cpu::kernels::quant_matmul::dequant_row_f32(src, dst_slice, format);
            }

            // Upload this chunk to the device at the correct byte offset.
            // Flatten f32 values to little-endian bytes without requiring bytemuck.
            let chunk_bytes: Vec<u8> = chunk_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
            let byte_offset = (row_start * row_k * 4) as u64;
            R::copy_to_device(&chunk_bytes, base_ptr + byte_offset, device).map_err(Error::Numr)?;

            // `chunk_f32` drops here, freeing the CPU buffer before the next chunk.
            row_start = row_end;
        }

        Ok(dst)
    }

    /// Load a quantized tensor as QuantTensor
    pub fn load_tensor_quantized<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<QuantTensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        let format = info
            .ggml_type
            .to_quant_format()
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor '{name}' type {:?} is not quantized", info.ggml_type),
            })?;

        let bytes = self.read_tensor_bytes(name)?;

        // GGUF stores shape in GGML order, reverse for row-major
        let mut shape = info.shape.clone();
        shape.reverse();

        QuantTensor::from_bytes(&bytes, format, &shape, device)
    }
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;
