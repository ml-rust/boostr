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

/// Backing storage for GGUF tensor data: either a regular file or a memory-mapped file.
enum GgufStorage {
    File(File),
    Mmap(Mmap),
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

    fn open_impl<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        let mut file = File::open(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let mut reader = BufReader::new(&mut file);

        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(Error::ModelError {
                reason: format!("invalid GGUF magic: 0x{magic:08x}"),
            });
        }

        let version = read_u32(&mut reader)?;
        if !(1..=3).contains(&version) {
            return Err(Error::ModelError {
                reason: format!("unsupported GGUF version: {version}"),
            });
        }

        let tensor_count = read_u64(&mut reader)?;
        let kv_count = read_u64(&mut reader)?;

        // Parse metadata
        let mut metadata = GgufMetadata::default();
        for _ in 0..kv_count {
            let (key, value) = read_kv_pair(&mut reader, version)?;
            metadata.kv.insert(key, value);
        }

        let alignment = metadata
            .get_u32("general.alignment")
            .map(|v| v as usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Parse tensor info entries
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = read_tensor_info(&mut reader, version)?;
            tensors.insert(info.name.clone(), info);
        }

        // Calculate aligned data offset
        let current_pos = reader.stream_position().map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let data_offset = align_offset(current_pos, alignment);

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

    /// Read raw tensor data bytes.
    ///
    /// When the file was opened with `use_mmap = true` this is a zero-copy slice
    /// of the mapping; otherwise it reads from the file.
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
