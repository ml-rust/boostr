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
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
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

    /// Load a tensor as F32 on the given device
    ///
    /// Currently supports loading F32 data. F16/BF16 data is converted to F32.
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

        let data: Vec<f32> = match info.dtype {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect(),
            DType::F64 => bytes
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            other => {
                return Err(Error::ModelError {
                    reason: format!("unsupported SafeTensors dtype: {other:?}"),
                });
            }
        };

        Tensor::<R>::try_from_slice(&data, &info.shape, device).map_err(Error::Numr)
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

/// Save tensors to SafeTensors format
///
/// Only accepts CPU tensors. Move GPU tensors to CPU before saving.
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Tensor<CpuRuntime>>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    use std::io::Write;

    let mut file = File::create(path).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    // Collect tensor data and build header
    let mut tensor_entries: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    for name in &names {
        let tensor = &tensors[*name];
        let data = tensor.to_vec::<f32>();
        let shape = tensor.shape().to_vec();
        tensor_entries.push(((*name).clone(), data, shape));
    }

    // Build header JSON
    let mut header = serde_json::Map::new();

    if let Some(meta) = metadata {
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in meta {
            meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(meta_obj));
    }

    let mut current_offset: usize = 0;
    for (name, data, shape) in &tensor_entries {
        let byte_len = data.len() * 4;
        let mut info = serde_json::Map::new();
        info.insert("dtype".into(), serde_json::Value::String("F32".into()));
        info.insert(
            "shape".into(),
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|&s| serde_json::Value::Number(s.into()))
                    .collect(),
            ),
        );
        info.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(current_offset.into()),
                serde_json::Value::Number((current_offset + byte_len).into()),
            ]),
        );
        header.insert(name.clone(), serde_json::Value::Object(info));
        current_offset += byte_len;
    }

    let header_str = serde_json::to_string(&serde_json::Value::Object(header)).map_err(|e| {
        Error::ModelError {
            reason: format!("JSON serialize error: {e}"),
        }
    })?;
    let header_bytes = header_str.as_bytes();

    // Write header size + header + data
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
    file.write_all(header_bytes)
        .map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
    for (_, data, _) in &tensor_entries {
        for f in data {
            file.write_all(&f.to_le_bytes())
                .map_err(|e| Error::ModelError {
                    reason: format!("IO error: {e}"),
                })?;
        }
    }
    file.flush().map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    Ok(())
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
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file() -> NamedTempFile {
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
    fn test_load_tensor_f32() {
        let (_, device) = cpu_setup();
        let f = create_test_file();
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("weight", &device).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_not_found() {
        let f = create_test_file();
        let st = SafeTensors::open(f.path()).unwrap();
        assert!(st.tensor_info("nonexistent").is_err());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let (_, device) = cpu_setup();
        let tmp = NamedTempFile::new().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "w1".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device),
        );

        save_safetensors(tmp.path(), &tensors, None).unwrap();

        let mut loaded = SafeTensors::open(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 1);
        let t = loaded.load_tensor::<CpuRuntime>("w1", &device).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        let data = t.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }
}
