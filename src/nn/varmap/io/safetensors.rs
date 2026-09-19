//! `VarMap` constructors reading SafeTensors files, single or sharded.

use super::super::core::VarMap;
use crate::error::{Error, Result};
use crate::format::safetensors::SafeTensors;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::HashMap;
use std::path::Path;

impl<R: Runtime<DType = DType>> VarMap<R> {
    /// Load all tensors from a SafeTensors file.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_with_model_type(path, device, None)
    }

    /// Load all tensors from a SafeTensors file, normalizing tensor names for
    /// the given model type (e.g., "falcon", "gpt_neox", "dbrx").
    pub fn from_safetensors_with_model_type<P: AsRef<Path>>(
        path: P,
        device: &R::Device,
        model_type: Option<&str>,
    ) -> Result<Self> {
        use crate::format::safetensors_name_map::normalize_hf_name;
        let mut st = SafeTensors::open(path)?;
        let all = st.load_all::<R>(device)?;
        let mut map = Self::new();
        for (name, tensor) in all {
            let mapped = match model_type {
                Some(mt) => normalize_hf_name(mt, &name),
                None => name,
            };
            map.insert(mapped, tensor);
        }
        Ok(map)
    }

    /// Load tensors from sharded SafeTensors files.
    ///
    /// Reads `model.safetensors.index.json` from `dir`, which maps tensor names
    /// to shard filenames (e.g., `model-00001-of-00004.safetensors`).
    /// Loads each shard once and extracts all its tensors.
    pub fn from_safetensors_sharded<P: AsRef<Path>>(dir: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_sharded_with_model_type(dir, device, None)
    }

    /// Load tensors from sharded SafeTensors files with name normalization.
    pub fn from_safetensors_sharded_with_model_type<P: AsRef<Path>>(
        dir: P,
        device: &R::Device,
        model_type: Option<&str>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path).map_err(|e| Error::ModelError {
            reason: format!("failed to read index file: {e}"),
        })?;

        let index: serde_json::Value =
            serde_json::from_str(&index_str).map_err(|e| Error::ModelError {
                reason: format!("failed to parse index JSON: {e}"),
            })?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::ModelError {
                reason: "index.json missing 'weight_map' object".into(),
            })?;

        // Group tensor names by shard file to open each shard only once
        let mut shard_to_names: HashMap<String, Vec<String>> = HashMap::new();
        for (tensor_name, shard_val) in weight_map {
            let shard_file = shard_val.as_str().ok_or_else(|| Error::ModelError {
                reason: format!("weight_map value for '{tensor_name}' is not a string"),
            })?;
            shard_to_names
                .entry(shard_file.to_string())
                .or_default()
                .push(tensor_name.clone());
        }

        let mut map = Self::new();

        for (shard_file, names) in &shard_to_names {
            let shard_path = dir.join(shard_file);
            let mut st = SafeTensors::open(&shard_path)?;
            for name in names {
                let tensor = st.load_tensor::<R>(name, device)?;
                let mapped = match model_type {
                    Some(mt) => crate::format::safetensors_name_map::normalize_hf_name(mt, name),
                    None => name.clone(),
                };
                map.insert(mapped, tensor);
            }
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::safetensors::save_safetensors;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_varmap_from_safetensors_roundtrip() {
        let d = device();
        let tmp = tempfile::NamedTempFile::new().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &d).unwrap(),
        );
        tensors.insert(
            "b".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &d).unwrap(),
        );
        save_safetensors(tmp.path(), &tensors, None).unwrap();

        let map = VarMap::<CpuRuntime>::from_safetensors(tmp.path(), &d).unwrap();
        assert_eq!(map.len(), 2);

        let a = map.get_tensor("a").unwrap();
        assert_eq!(a.shape(), &[3]);
        let data = a.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);

        let b = map.get_tensor("b").unwrap();
        assert_eq!(b.shape(), &[2]);
    }

    #[test]
    fn test_varmap_from_safetensors_sharded() {
        let d = device();
        let dir = tempfile::TempDir::new().unwrap();

        // Create two shard files
        let mut shard1 = HashMap::new();
        shard1.insert(
            "layers.0.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &d).unwrap(),
        );
        save_safetensors(
            dir.path().join("model-00001-of-00002.safetensors"),
            &shard1,
            None,
        )
        .unwrap();

        let mut shard2 = HashMap::new();
        shard2.insert(
            "layers.1.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &d).unwrap(),
        );
        save_safetensors(
            dir.path().join("model-00002-of-00002.safetensors"),
            &shard2,
            None,
        )
        .unwrap();

        // Create index.json
        let index = serde_json::json!({
            "metadata": {"total_size": 20},
            "weight_map": {
                "layers.0.weight": "model-00001-of-00002.safetensors",
                "layers.1.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();

        // Load sharded
        let map = VarMap::<CpuRuntime>::from_safetensors_sharded(dir.path(), &d).unwrap();
        assert_eq!(map.len(), 2);

        let t0 = map.get_tensor("layers.0.weight").unwrap();
        assert_eq!(t0.shape(), &[2]);
        let data0: Vec<f32> = t0.to_vec();
        assert!((data0[0] - 1.0).abs() < 1e-6);

        let t1 = map.get_tensor("layers.1.weight").unwrap();
        assert_eq!(t1.shape(), &[3]);
        let data1: Vec<f32> = t1.to_vec();
        assert!((data1[2] - 5.0).abs() < 1e-6);
    }
}
