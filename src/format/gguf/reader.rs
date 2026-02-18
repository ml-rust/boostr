//! GGUF file reader
//!
//! Parses the GGUF binary format (v1-v3) and provides access to metadata
//! and tensor data. F32 tensors are loaded as `Tensor<R>`, quantized tensors
//! as `QuantTensor<R>`.

use super::GgufTensorInfo;
use super::metadata::GgufMetadata;
use super::types::{GgmlType, GgufValueType};
use super::value::GgufValue;
use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGUF file reader
pub struct Gguf {
    file: File,
    version: u32,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensorInfo>,
    data_offset: u64,
}

impl Gguf {
    /// Open and parse a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
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

        Ok(Gguf {
            file,
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

    /// Read raw tensor data bytes
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
        let mut buf = vec![0u8; size];

        self.file
            .seek(SeekFrom::Start(abs_offset))
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
                return Err(Error::ModelError {
                    reason: format!(
                        "tensor '{name}' has quantized type {other:?}, use load_tensor_quantized"
                    ),
                });
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

// ── Binary reading helpers ──────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R, _version: u32) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    String::from_utf8(buf).map_err(|e| Error::ModelError {
        reason: format!("invalid UTF-8: {e}"),
    })
}

fn read_value<R: Read>(r: &mut R, vt: GgufValueType, version: u32) -> Result<GgufValue> {
    match vt {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(read_u8(r)?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(read_u8(r)? as i8)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(read_u16(r)?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(read_u16(r)? as i16)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(read_u32(r)?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(read_i32(r)?)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(read_f32(r)?)),
        GgufValueType::Bool => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GgufValueType::String => Ok(GgufValue::String(read_string(r, version)?)),
        GgufValueType::Array => {
            let elem_type = read_u32(r)?;
            let elem_type =
                GgufValueType::from_u32(elem_type).ok_or_else(|| Error::ModelError {
                    reason: format!("invalid array element type: {elem_type}"),
                })?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(r, elem_type, version)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(read_u64(r)?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(read_i64(r)?)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(read_f64(r)?)),
    }
}

fn read_kv_pair<R: Read>(r: &mut R, version: u32) -> Result<(String, GgufValue)> {
    let key = read_string(r, version)?;
    let vt_raw = read_u32(r)?;
    let vt = GgufValueType::from_u32(vt_raw).ok_or_else(|| Error::ModelError {
        reason: format!("invalid value type: {vt_raw}"),
    })?;
    let value = read_value(r, vt, version)?;
    Ok((key, value))
}

fn read_tensor_info<R: Read>(r: &mut R, version: u32) -> Result<GgufTensorInfo> {
    let name = read_string(r, version)?;
    let n_dims = read_u32(r)?;

    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(read_u64(r)? as usize);
    }

    let ggml_type_raw = read_u32(r)?;
    let ggml_type = GgmlType::from_u32(ggml_type_raw).ok_or_else(|| Error::ModelError {
        reason: format!("unsupported GGML type: {ggml_type_raw}"),
    })?;

    let offset = read_u64(r)?;

    Ok(GgufTensorInfo {
        name,
        n_dims,
        shape,
        ggml_type,
        offset,
    })
}

fn align_offset(offset: u64, alignment: usize) -> u64 {
    let a = alignment as u64;
    offset.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: write a GGUF string (u64 length + bytes)
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Create a minimal GGUF v3 file with one F32 tensor and one Q4_0 tensor
    fn create_test_gguf() -> NamedTempFile {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // 2 tensors
        buf.extend_from_slice(&2u64.to_le_bytes());
        // 2 KV pairs
        buf.extend_from_slice(&2u64.to_le_bytes());

        // KV 1: general.architecture = "test"
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_str(&mut buf, "test");

        // KV 2: test.block_count = 4
        write_str(&mut buf, "test.block_count");
        buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        // Tensor 1: "weight_f32" F32 [4]
        write_str(&mut buf, "weight_f32");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Tensor 2: "weight_q4" Q4_0 [32]
        write_str(&mut buf, "weight_q4");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::Q4_0 as u32).to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes()); // offset after 4 floats

        // Align to 32 bytes
        let aligned = buf.len().div_ceil(32) * 32;
        buf.resize(aligned, 0);

        // Data: weight_f32 = [1.0, 2.0, 3.0, 4.0]
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }

        // Data: weight_q4 - Q4_0 block (scale=1.0, all nibbles=8 -> dequant to 0)
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        buf.push((scale_bits & 0xFF) as u8);
        buf.push(((scale_bits >> 8) & 0xFF) as u8);
        buf.extend(std::iter::repeat_n(0x88u8, 16));

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_open_gguf() {
        let f = create_test_gguf();
        let gguf = Gguf::open(f.path()).unwrap();
        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.len(), 2);
        assert_eq!(gguf.metadata().architecture(), Some("test"));
        assert_eq!(gguf.metadata().block_count(), Some(4));
    }

    #[test]
    fn test_tensor_info_gguf() {
        let f = create_test_gguf();
        let gguf = Gguf::open(f.path()).unwrap();

        let f32_info = gguf.tensor_info("weight_f32").unwrap();
        assert_eq!(f32_info.shape, vec![4]);
        assert_eq!(f32_info.ggml_type, GgmlType::F32);
        assert_eq!(f32_info.size_bytes(), 16);

        let q4_info = gguf.tensor_info("weight_q4").unwrap();
        assert_eq!(q4_info.shape, vec![32]);
        assert_eq!(q4_info.ggml_type, GgmlType::Q4_0);
        assert_eq!(q4_info.size_bytes(), 18);
    }

    #[test]
    fn test_load_f32_tensor() {
        let (_, device) = cpu_setup();
        let f = create_test_gguf();
        let mut gguf = Gguf::open(f.path()).unwrap();

        let tensor = gguf
            .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
            .unwrap();
        // GGUF 1D tensor: shape reversed is still [4]
        assert_eq!(tensor.shape(), &[4]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_quantized_tensor() {
        let f = create_test_gguf();
        let mut gguf = Gguf::open(f.path()).unwrap();

        let device = numr::runtime::cpu::CpuDevice::new();
        let qt = gguf
            .load_tensor_quantized::<numr::runtime::cpu::CpuRuntime>("weight_q4", &device)
            .unwrap();
        assert_eq!(qt.shape(), &[32]);
        assert_eq!(qt.format(), crate::quant::QuantFormat::Q4_0);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_tensor_not_found() {
        let f = create_test_gguf();
        let gguf = Gguf::open(f.path()).unwrap();
        assert!(gguf.tensor_info("nonexistent").is_err());
    }
}
