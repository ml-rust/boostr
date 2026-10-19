//! SafeTensors file writer

use super::encode::tensor_to_le_bytes;
use super::header::{Spec, encode_header};
use crate::error::{Error, Result};
use crate::format::atomic_write::write_atomic;
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

/// Save tensors to SafeTensors format
///
/// Each tensor is written at its own dtype — a BF16 tensor produces a BF16
/// entry at 2 bytes per element, not an F32 one at 4.
///
/// The function builds the header from shapes and dtypes before it touches
/// the filesystem. It then writes a sibling temp file, one encoded tensor at
/// a time, and renames it over `path`. Peak memory tracks the largest tensor.
/// An error leaves any existing file at `path` unchanged.
///
/// Only accepts CPU tensors. Move GPU tensors to CPU before saving.
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Tensor<CpuRuntime>>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let shapes: Vec<&[usize]> = names.iter().map(|n| tensors[*n].shape()).collect();
    let specs: Vec<Spec<'_>> = names
        .iter()
        .zip(&shapes)
        .map(|(n, shape)| (n.as_str(), *shape, tensors[*n].dtype()))
        .collect();
    let header = encode_header(&specs, metadata)?;

    write_atomic(path, |w| {
        w.write_all(&header)?;
        for name in &names {
            let data = tensor_to_le_bytes(&tensors[*name])
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            w.write_all(&data)?;
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
    use numr::dtype::DType;
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

    /// Reference encoder: encodes every tensor into one buffer, then the header.
    fn reference_save(
        tensors: &HashMap<String, Tensor<CpuRuntime>>,
        metadata: Option<&HashMap<String, String>>,
    ) -> Vec<u8> {
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        let mut header = serde_json::Map::new();
        if let Some(meta) = metadata {
            let mut meta_obj = serde_json::Map::new();
            for (k, v) in meta {
                meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            header.insert("__metadata__".into(), serde_json::Value::Object(meta_obj));
        }
        let mut body = Vec::new();
        for name in names {
            let t = &tensors[name];
            let data = tensor_to_le_bytes(t).unwrap();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": crate::format::safetensors::save::encode::dtype_name(t.dtype()).unwrap(),
                    "shape": t.shape(),
                    "data_offsets": [body.len(), body.len() + data.len()],
                }),
            );
            body.extend_from_slice(&data);
        }
        let mut header = serde_json::Value::Object(header);
        header.sort_all_objects();
        let header_str = serde_json::to_string(&header).unwrap();
        let mut out = (header_str.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header_str.as_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn tensor_of(
        bytes: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &numr::runtime::cpu::CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let storage =
            numr::tensor::Storage::<CpuRuntime>::from_bytes(bytes, dtype, device).unwrap();
        Tensor::<CpuRuntime>::from_storage_contiguous(storage, shape)
    }

    /// Streamed output must be byte-identical to the reference encoder.
    #[test]
    fn test_streamed_bytes_match_reference_encoder() {
        let (_, device) = cpu_setup();
        let mut tensors = HashMap::new();
        tensors.insert(
            "b.f32".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[7.5f32, -1.25, 0.0], &[3], &device).unwrap(),
        );
        tensors.insert(
            "a.f32".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
        );
        tensors.insert(
            "c.i64".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[-5i64, 0, 9], &[3], &device).unwrap(),
        );
        tensors.insert(
            "d.scalar".to_string(),
            tensor_of(&2.5f32.to_le_bytes(), DType::F32, &[], &device),
        );
        tensors.insert(
            "e.empty".to_string(),
            tensor_of(&[], DType::F32, &[0], &device),
        );
        #[cfg(feature = "f16")]
        {
            let h: Vec<u8> = [1.0f32, 2.5, -3.0]
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect();
            let b: Vec<u8> = [1.0f32, 2.5, -3.0]
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            tensors.insert(
                "f.f16".to_string(),
                tensor_of(&h, DType::F16, &[3], &device),
            );
            tensors.insert(
                "g.bf16".to_string(),
                tensor_of(&b, DType::BF16, &[3], &device),
            );
        }
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "pt".to_string());

        let tmp = NamedTempFile::new().unwrap();
        save_safetensors(tmp.path(), &tensors, Some(&metadata)).unwrap();

        let actual = std::fs::read(tmp.path()).unwrap();
        assert_eq!(actual, reference_save(&tensors, Some(&metadata)));
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
