//! SafeTensors header parsing, tensor metadata, and raw byte reads.

use crate::error::{Error, Result};
use numr::dtype::DType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

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
    pub(super) tensors: HashMap<String, TensorInfo>,
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
}

pub(super) fn parse_dtype(s: &str) -> Result<DType> {
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
pub(super) mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    pub(in super::super) fn create_test_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();

        let header = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24]
            }
        });
        let header_str = header.to_string();
        let header_bytes = header_str.as_bytes();

        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_bytes).unwrap();

        for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            file.write_all(&f.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_open_and_metadata() {
        let f = create_test_file();
        let st = SafeTensors::open(f.path()).unwrap();
        assert_eq!(st.len(), 1);
        assert_eq!(st.metadata().get("format"), Some(&"pt".to_string()));
    }

    #[test]
    fn test_tensor_info() {
        let f = create_test_file();
        let st = SafeTensors::open(f.path()).unwrap();
        let info = st.tensor_info("weight").unwrap();
        assert_eq!(info.dtype, DType::F32);
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.numel(), 6);
        assert_eq!(info.size_bytes(), 24);
    }

    #[test]
    fn test_tensor_not_found() {
        let f = create_test_file();
        let st = SafeTensors::open(f.path()).unwrap();
        assert!(st.tensor_info("nonexistent").is_err());
    }

    /// An unsupported dtype string must fail with an `Err` naming the dtype,
    /// never a panic — checked at `SafeTensors::open`, where `parse_dtype` runs.
    #[test]
    fn test_load_tensor_unsupported_dtype_names_it() {
        let mut file = NamedTempFile::new().unwrap();
        let header = serde_json::json!({
            "w": { "dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1] }
        });
        let header_str = header.to_string();
        let header_bytes = header_str.as_bytes();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_bytes).unwrap();
        file.write_all(&[0u8]).unwrap();
        file.flush().unwrap();

        // `unwrap_err` would require `SafeTensors: Debug`; match instead of
        // widening the public type's derives just to satisfy a test.
        let msg = match SafeTensors::open(file.path()) {
            Ok(_) => panic!("expected an error naming the unsupported dtype"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("F8_E4M3"),
            "error should name the dtype: {msg}"
        );
    }
}
