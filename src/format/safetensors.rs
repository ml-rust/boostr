//! SafeTensors file format parser and loader
//!
//! SafeTensors is a simple, safe format for storing tensors developed by HuggingFace.
//!
//! # Format
//!
//! ```text
//! [8 bytes] header_size (little-endian u64)
//! [header_size bytes] JSON header containing:
//!   - "__metadata__": optional dict of string key-value pairs
//!   - "<tensor_name>": { "dtype": str, "shape": [int], "data_offsets": [start, end] }
//! [remaining bytes] raw tensor data
//! ```

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

mod save;
pub use save::save_safetensors;

/// Information about a tensor in a SafeTensors file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.data_end - self.data_start
    }
}

/// SafeTensors file reader
pub struct SafeTensors {
    file: File,
    data_offset: u64,
    tensors: HashMap<String, TensorInfo>,
    metadata: HashMap<String, String>,
}

impl SafeTensors {
    /// Open and parse a SafeTensors file header
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let file_size = file
            .metadata()
            .map_err(|e| Error::ModelError {
                reason: format!("IO error: {e}"),
            })?
            .len();

        // Read header size (8 bytes, little-endian u64)
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).map_err(|e| Error::ModelError {
            reason: format!("IO error reading header size: {e}"),
        })?;
        let header_size = u64::from_le_bytes(buf);

        if header_size > file_size - 8 {
            return Err(Error::ModelError {
                reason: format!("header size {header_size} exceeds file size {file_size}"),
            });
        }

        // Read and parse header JSON
        let mut header_buf = vec![0u8; header_size as usize];
        file.read_exact(&mut header_buf)
            .map_err(|e| Error::ModelError {
                reason: format!("IO error reading header: {e}"),
            })?;

        let header_str = std::str::from_utf8(&header_buf).map_err(|e| Error::ModelError {
            reason: format!("invalid UTF-8 in header: {e}"),
        })?;

        let header: serde_json::Value =
            serde_json::from_str(header_str).map_err(|e| Error::ModelError {
                reason: format!("JSON parse error: {e}"),
            })?;

        let header_obj = header.as_object().ok_or_else(|| Error::ModelError {
            reason: "header is not a JSON object".into(),
        })?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        for (key, value) in header_obj {
            if key == "__metadata__" {
                if let Some(meta_obj) = value.as_object() {
                    for (mk, mv) in meta_obj {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
            } else {
                let obj = value.as_object().ok_or_else(|| Error::ModelError {
                    reason: format!("tensor '{key}' is not an object"),
                })?;

                let dtype_str =
                    obj.get("dtype")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::ModelError {
                            reason: format!("tensor '{key}' missing dtype"),
                        })?;

                let dtype = parse_dtype(dtype_str)?;

                let shape: Vec<usize> = obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| Error::ModelError {
                        reason: format!("tensor '{key}' missing shape"),
                    })?
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect();

                let offsets = obj
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| Error::ModelError {
                        reason: format!("tensor '{key}' missing data_offsets"),
                    })?;

                if offsets.len() != 2 {
                    return Err(Error::ModelError {
                        reason: format!("tensor '{key}' data_offsets must have 2 elements"),
                    });
                }

                let data_start = offsets[0].as_u64().ok_or_else(|| Error::ModelError {
                    reason: format!("tensor '{key}' invalid data_offsets[0]"),
                })? as usize;

                let data_end = offsets[1].as_u64().ok_or_else(|| Error::ModelError {
                    reason: format!("tensor '{key}' invalid data_offsets[1]"),
                })? as usize;

                tensors.insert(
                    key.clone(),
                    TensorInfo {
                        name: key.clone(),
                        dtype,
                        shape,
                        data_start,
                        data_end,
                    },
                );
            }
        }

        Ok(SafeTensors {
            file,
            data_offset: 8 + header_size,
            tensors,
            metadata,
        })
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

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::ModelError {
            reason: format!("tensor not found: {name}"),
        })
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Read raw tensor data as bytes
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor not found: {name}"),
            })?
            .clone();

        let abs_start = self.data_offset + info.data_start as u64;
        let size = info.size_bytes();
        let mut buf = vec![0u8; size];

        self.file
            .seek(SeekFrom::Start(abs_start))
            .map_err(|e| Error::ModelError {
                reason: format!("IO seek error: {e}"),
            })?;
        self.file
            .read_exact(&mut buf)
            .map_err(|e| Error::ModelError {
                reason: format!("IO read error: {e}"),
            })?;

        Ok(buf)
    }

    /// Load a tensor in its native dtype on the given device
    ///
    /// Preserves the original dtype from the SafeTensors file (F32, F16, BF16, etc.)
    /// without converting to F32. This halves memory for BF16/F16 models.
    pub fn load_tensor<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor not found: {name}"),
            })?
            .clone();

        let bytes = self.read_tensor_bytes(name)?;

        match info.dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                // Load raw bytes directly in native dtype
                let storage = numr::tensor::Storage::<R>::from_bytes(&bytes, info.dtype, device)
                    .map_err(Error::Numr)?;
                Ok(Tensor::<R>::from_storage_contiguous(storage, &info.shape))
            }
            DType::F64 => {
                // Downcast F64 to F32 (F64 weights are rare and wasteful)
                let data: Vec<f32> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| {
                        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                    })
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::I64 => {
                // SafeTensors stores integers little-endian; decode explicitly
                // rather than reinterpreting raw bytes, so this is correct on
                // both little- and big-endian hosts.
                let data: Vec<i64> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::I32 => {
                // Little-endian decode, see the I64 arm above.
                let data: Vec<i32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| i32::from_le_bytes(*b))
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::U32 => {
                // Little-endian decode, see the I64 arm above.
                let data: Vec<u32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::I8 | DType::Bool => {
                // Single-byte types have no endianness to decode; load the
                // raw bytes directly, same as the float arm above.
                let storage = numr::tensor::Storage::<R>::from_bytes(&bytes, info.dtype, device)
                    .map_err(Error::Numr)?;
                Ok(Tensor::<R>::from_storage_contiguous(storage, &info.shape))
            }
            // Reached only if `parse_dtype` is ever extended to accept a
            // SafeTensors dtype string this arm doesn't yet decode (e.g.
            // I16/U16/U64/U8, which numr's DType supports but this loader
            // does not yet handle) or DType gains a variant `parse_dtype`
            // can never produce (Complex64/128, FP8). Names the dtype so
            // the caller sees what to add rather than silently coercing it.
            other => Err(Error::ModelError {
                reason: format!("unsupported SafeTensors dtype: {other:?}"),
            }),
        }
    }

    /// Load all tensors to the given device
    pub fn load_all<R: Runtime<DType = DType>>(
        &mut self,
        device: &R::Device,
    ) -> Result<HashMap<String, Tensor<R>>> {
        let names: Vec<String> = self.tensors.keys().cloned().collect();
        let mut result = HashMap::with_capacity(names.len());
        for name in names {
            let tensor = self.load_tensor::<R>(&name, device)?;
            result.insert(name, tensor);
        }
        Ok(result)
    }
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s {
        "F32" | "f32" | "float32" => Ok(DType::F32),
        "F16" | "f16" | "float16" => Ok(DType::F16),
        "BF16" | "bf16" | "bfloat16" => Ok(DType::BF16),
        "F64" | "f64" | "float64" => Ok(DType::F64),
        "I32" | "i32" | "int32" => Ok(DType::I32),
        "I64" | "i64" | "int64" => Ok(DType::I64),
        "U32" | "u32" | "uint32" => Ok(DType::U32),
        "I8" | "i8" | "int8" => Ok(DType::I8),
        "BOOL" | "bool" => Ok(DType::Bool),
        _ => Err(Error::ModelError {
            reason: format!("unsupported SafeTensors dtype: {s}"),
        }),
    }
}

#[cfg(test)]
mod tests;
