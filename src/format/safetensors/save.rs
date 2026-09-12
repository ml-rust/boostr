//! SafeTensors file format writer

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Encode host values as little-endian bytes
///
/// `to_le` fixes the per-element width, so the byte length is always
/// `data.len() * N` — never a hard-wired constant.
fn encode_le<T, const N: usize>(data: Vec<T>, to_le: fn(T) -> [u8; N]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * N);
    for v in data {
        out.extend_from_slice(&to_le(v));
    }
    out
}

/// Copy a CPU tensor to host bytes at its OWN dtype width
///
/// Each arm reads with a host type whose width equals the tensor's dtype width.
/// Reading at a wider type (e.g. `to_vec::<f32>()` on a BF16 tensor) copies more
/// bytes than the storage holds, so it must never happen.
fn tensor_to_le_bytes(tensor: &Tensor<CpuRuntime>) -> Result<Vec<u8>> {
    let bytes = match tensor.dtype() {
        DType::F32 => encode_le(tensor.to_vec::<f32>(), f32::to_le_bytes),
        DType::F64 => encode_le(tensor.to_vec::<f64>(), f64::to_le_bytes),
        DType::F16 => encode_le(tensor.to_vec::<half::f16>(), half::f16::to_le_bytes),
        DType::BF16 => encode_le(tensor.to_vec::<half::bf16>(), half::bf16::to_le_bytes),
        DType::I64 => encode_le(tensor.to_vec::<i64>(), i64::to_le_bytes),
        DType::I32 => encode_le(tensor.to_vec::<i32>(), i32::to_le_bytes),
        DType::I8 => encode_le(tensor.to_vec::<i8>(), i8::to_le_bytes),
        DType::U32 => encode_le(tensor.to_vec::<u32>(), u32::to_le_bytes),
        // Bool is stored one byte per element, same as u8.
        DType::Bool => encode_le(tensor.to_vec::<u8>(), u8::to_le_bytes),
        other => {
            return Err(Error::ModelError {
                reason: format!("unsupported SafeTensors save dtype: {other:?}"),
            });
        }
    };
    Ok(bytes)
}

/// SafeTensors header name for a dtype
///
/// Mirrors [`parse_dtype`](super::header::parse_dtype) exactly: boostr writes every
/// dtype it can read.
fn dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("F32"),
        DType::F16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::F64 => Ok("F64"),
        DType::I32 => Ok("I32"),
        DType::I64 => Ok("I64"),
        DType::U32 => Ok("U32"),
        DType::I8 => Ok("I8"),
        DType::Bool => Ok("BOOL"),
        other => Err(Error::ModelError {
            reason: format!("unsupported SafeTensors save dtype: {other:?}"),
        }),
    }
}

/// Save tensors to SafeTensors format
///
/// Each tensor is written at its own dtype — a BF16 tensor produces a BF16
/// entry at 2 bytes per element, not an F32 one at 4.
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
    let mut tensor_entries: Vec<(String, Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    for name in &names {
        let tensor = &tensors[*name];
        let dtype = tensor.dtype();
        let data = tensor_to_le_bytes(tensor)?;
        let shape = tensor.shape().to_vec();
        tensor_entries.push(((*name).clone(), data, shape, dtype));
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
    for (name, data, shape, dtype) in &tensor_entries {
        let byte_len = data.len();
        let mut info = serde_json::Map::new();
        info.insert(
            "dtype".into(),
            serde_json::Value::String(dtype_name(*dtype)?.into()),
        );
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
    for (_, data, _, _) in &tensor_entries {
        file.write_all(data).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
    }
    file.flush().map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::SafeTensors;
    use super::*;
    use crate::test_utils::cpu_setup;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_and_load_roundtrip() {
        let (_, device) = cpu_setup();
        let tmp = NamedTempFile::new().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "w1".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
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

    // ===== Save path: per-dtype writing =====

    /// Every dtype `parse_dtype` reads, `save_safetensors` must write.
    const SAVE_DTYPES: [DType; 9] = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::I8,
        DType::Bool,
    ];

    const SAMPLE_VALUES: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    /// Little-endian bytes for `SAMPLE_VALUES` encoded at `dtype`'s own width.
    fn sample_bytes(dtype: DType) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, &v) in SAMPLE_VALUES.iter().enumerate() {
            match dtype {
                DType::F32 => out.extend_from_slice(&v.to_le_bytes()),
                DType::F64 => out.extend_from_slice(&f64::from(v).to_le_bytes()),
                DType::F16 => out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                DType::BF16 => out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
                DType::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
                DType::I8 => out.extend_from_slice(&(v as i8).to_le_bytes()),
                DType::U32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
                DType::Bool => out.push((i % 2) as u8),
                other => panic!("unsupported test dtype: {other:?}"),
            }
        }
        out
    }

    fn cpu_tensor_from_bytes(
        bytes: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &numr::runtime::cpu::CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let storage =
            numr::tensor::Storage::<CpuRuntime>::from_bytes(bytes, dtype, device).unwrap();
        Tensor::<CpuRuntime>::from_storage_contiguous(storage, shape)
    }

    /// Save a single 2x3 tensor of `dtype` built from `sample_bytes`.
    fn save_sample(
        dtype: DType,
        device: &numr::runtime::cpu::CpuDevice,
    ) -> (NamedTempFile, Vec<u8>) {
        let bytes = sample_bytes(dtype);
        let tensor = cpu_tensor_from_bytes(&bytes, dtype, &[2, 3], device);
        let mut tensors = HashMap::new();
        tensors.insert("w".to_string(), tensor);
        let tmp = NamedTempFile::new().unwrap();
        save_safetensors(tmp.path(), &tensors, None).unwrap();
        (tmp, bytes)
    }

    fn read_header_json(path: &std::path::Path) -> serde_json::Value {
        let raw = std::fs::read(path).unwrap();
        let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        serde_json::from_slice(&raw[8..8 + n]).unwrap()
    }

    /// On-disk bytes per element must equal the dtype's width.
    ///
    /// This is the assertion that catches a silent re-widening to F32: before the
    /// per-dtype write path, a BF16 tensor was saved as 4 bytes/element with an
    /// "F32" header, so `size_bytes()` here was 24 instead of 12 — and the host
    /// read itself was an out-of-bounds `to_vec::<f32>()` over 12 bytes of storage.
    #[test]
    fn test_save_writes_native_element_width() {
        let (_, device) = cpu_setup();
        for dtype in SAVE_DTYPES {
            let (tmp, _) = save_sample(dtype, &device);
            let st = SafeTensors::open(tmp.path()).unwrap();
            let info = st.tensor_info("w").unwrap();
            assert_eq!(info.dtype, dtype, "dtype mismatch for {dtype:?}");
            assert_eq!(
                info.size_bytes(),
                6 * dtype.size_in_bytes(),
                "byte width mismatch for {dtype:?}"
            );
            // The data section holds exactly one payload, at the native width.
            let raw = std::fs::read(tmp.path()).unwrap();
            let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
            assert_eq!(
                raw.len(),
                8 + header_len + 6 * dtype.size_in_bytes(),
                "file length mismatch for {dtype:?}"
            );
        }
    }

    /// The header's `dtype` string must name the tensor's own dtype.
    #[test]
    fn test_save_header_dtype_string() {
        let (_, device) = cpu_setup();
        let expected = [
            (DType::F32, "F32"),
            (DType::F16, "F16"),
            (DType::BF16, "BF16"),
            (DType::F64, "F64"),
            (DType::I32, "I32"),
            (DType::I64, "I64"),
            (DType::U32, "U32"),
            (DType::I8, "I8"),
            (DType::Bool, "BOOL"),
        ];
        for (dtype, name) in expected {
            let (tmp, _) = save_sample(dtype, &device);
            let header = read_header_json(tmp.path());
            assert_eq!(header["w"]["dtype"].as_str(), Some(name), "for {dtype:?}");
            assert_eq!(
                header["w"]["data_offsets"][1].as_u64(),
                Some((6 * dtype.size_in_bytes()) as u64),
                "data_offsets width for {dtype:?}"
            );
        }
    }

    /// Raw payload bytes survive the save unchanged, for every supported dtype.
    #[test]
    fn test_save_roundtrip_raw_bytes() {
        let (_, device) = cpu_setup();
        for dtype in SAVE_DTYPES {
            let (tmp, original) = save_sample(dtype, &device);
            let mut st = SafeTensors::open(tmp.path()).unwrap();
            assert_eq!(
                st.read_tensor_bytes("w").unwrap(),
                original,
                "payload mismatch for {dtype:?}"
            );
        }
    }

    /// Values and dtype both survive a save/reload through `load_tensor`.
    #[test]
    fn test_save_roundtrip_preserves_float_dtype() {
        let (_, device) = cpu_setup();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let (tmp, _) = save_sample(dtype, &device);
            let mut st = SafeTensors::open(tmp.path()).unwrap();
            let t = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.dtype(), dtype, "reloaded dtype changed for {dtype:?}");

            let got: Vec<f32> = match dtype {
                DType::F32 => t.to_vec::<f32>(),
                DType::F16 => t.to_vec::<half::f16>().iter().map(|v| v.to_f32()).collect(),
                _ => t
                    .to_vec::<half::bf16>()
                    .iter()
                    .map(|v| v.to_f32())
                    .collect(),
            };
            for (g, v) in got.iter().zip(SAMPLE_VALUES) {
                assert!((g - v).abs() < 1e-2, "value mismatch for {dtype:?}");
            }
        }
    }

    /// Byte-for-byte reimplementation of the pre-change F32-only writer.
    fn legacy_save_f32(
        tensors: &HashMap<String, Tensor<CpuRuntime>>,
        metadata: Option<&HashMap<String, String>>,
    ) -> Vec<u8> {
        let mut entries: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        for name in &names {
            let tensor = &tensors[*name];
            entries.push((
                (*name).clone(),
                tensor.to_vec::<f32>(),
                tensor.shape().to_vec(),
            ));
        }

        let mut header = serde_json::Map::new();
        if let Some(meta) = metadata {
            let mut meta_obj = serde_json::Map::new();
            for (k, v) in meta {
                meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            header.insert("__metadata__".into(), serde_json::Value::Object(meta_obj));
        }

        let mut current_offset: usize = 0;
        for (name, data, shape) in &entries {
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

        let header_str = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        out.extend_from_slice(header_str.as_bytes());
        for (_, data, _) in &entries {
            for f in data {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        out
    }

    /// An F32 file must be byte-identical to what the old writer produced.
    #[test]
    fn test_save_f32_bytes_unchanged() {
        let (_, device) = cpu_setup();
        let mut tensors = HashMap::new();
        tensors.insert(
            "b.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[7.5f32, -1.25, 0.0], &[3], &device).unwrap(),
        );
        tensors.insert(
            "a.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
        );
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "pt".to_string());

        let tmp = NamedTempFile::new().unwrap();
        save_safetensors(tmp.path(), &tensors, Some(&metadata)).unwrap();

        let actual = std::fs::read(tmp.path()).unwrap();
        let expected = legacy_save_f32(&tensors, Some(&metadata));
        assert_eq!(actual, expected, "F32 output changed");
    }
}
