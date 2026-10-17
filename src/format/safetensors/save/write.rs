//! SafeTensors file writer

use super::encode::{dtype_name, tensor_to_le_bytes};
use crate::error::{Error, Result};
use crate::format::atomic_write::write_atomic;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn io_error(e: std::io::Error) -> Error {
    Error::ModelError {
        reason: format!("IO error: {e}"),
    }
}

/// Serialize the header JSON, prefixed by its little-endian `u64` length
///
/// Runs before any file is touched, so an unsupported dtype or a
/// serialization error leaves the filesystem unchanged.
fn encode_header(
    entries: &[(String, Vec<u8>, Vec<usize>, DType)],
    metadata: Option<&HashMap<String, String>>,
) -> Result<Vec<u8>> {
    let mut header = serde_json::Map::new();

    if let Some(meta) = metadata {
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in meta {
            meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(meta_obj));
    }

    let mut current_offset: usize = 0;
    for (name, data, shape, dtype) in entries {
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
    let mut out = Vec::with_capacity(8 + header_str.len());
    out.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    out.extend_from_slice(header_str.as_bytes());
    Ok(out)
}

/// Save tensors to SafeTensors format
///
/// Each tensor is written at its own dtype — a BF16 tensor produces a BF16
/// entry at 2 bytes per element, not an F32 one at 4.
///
/// The function encodes every tensor and the header before it touches the
/// filesystem. It then writes a sibling temp file and renames it over `path`.
/// An error leaves any existing file at `path` unchanged.
///
/// Only accepts CPU tensors. Move GPU tensors to CPU before saving.
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Tensor<CpuRuntime>>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    let mut entries: Vec<(String, Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    for name in &names {
        let tensor = &tensors[*name];
        let data = tensor_to_le_bytes(tensor)?;
        entries.push((
            (*name).clone(),
            data,
            tensor.shape().to_vec(),
            tensor.dtype(),
        ));
    }

    let header = encode_header(&entries, metadata)?;

    write_atomic(path, |w| {
        w.write_all(&header)?;
        for (_, data, _, _) in &entries {
            w.write_all(data)?;
        }
        Ok(())
    })
    .map_err(io_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::safetensors::SafeTensors;
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

    /// Independent reference encoder for F32 tensors.
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

    /// An F32 file must be byte-identical to the reference encoder output.
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
        assert_eq!(actual, expected, "F32 output differs from reference");
    }

    fn temp_files_in(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect()
    }

    /// An unsupported dtype fails before the filesystem is touched.
    #[test]
    fn test_unsupported_dtype_keeps_existing_file() {
        let (_, device) = cpu_setup();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let mut good = HashMap::new();
        good.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap(),
        );
        save_safetensors(&path, &good, None).unwrap();
        let before = std::fs::read(&path).unwrap();

        let storage = numr::tensor::Storage::<CpuRuntime>::from_bytes(
            &7u64.to_le_bytes(),
            DType::U64,
            &device,
        )
        .unwrap();
        let mut bad = HashMap::new();
        bad.insert(
            "x".to_string(),
            Tensor::<CpuRuntime>::from_storage_contiguous(storage, &[1]),
        );
        assert!(save_safetensors(&path, &bad, None).is_err());

        assert_eq!(std::fs::read(&path).unwrap(), before);
        let mut loaded = SafeTensors::open(&path).unwrap();
        let t = loaded.load_tensor::<CpuRuntime>("w", &device).unwrap();
        assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
        assert!(temp_files_in(dir.path()).is_empty());
    }

    /// A successful save leaves only the target file in the directory.
    #[test]
    fn test_successful_save_leaves_no_temp_file() {
        let (_, device) = cpu_setup();
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
        );
        save_safetensors(dir.path().join("m.safetensors"), &tensors, None).unwrap();
        save_safetensors(dir.path().join("m.safetensors"), &tensors, None).unwrap();

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["m.safetensors".to_string()]);
    }
}
