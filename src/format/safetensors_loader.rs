//! Multi-file SafeTensors loader
//!
//! Provides unified loading for single-file and sharded (multi-file) models.

use crate::error::{Error, Result};
use crate::format::safetensors::{SafeTensors, TensorInfo};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

/// Unified multi-file SafeTensors loader
///
/// Handles both single-file and sharded models transparently.
pub struct SafeTensorsLoader {
    files: Vec<SafeTensors>,
    model_dir: PathBuf,
}

impl SafeTensorsLoader {
    /// Open a model from a directory or single file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if path.is_file() {
            let st = SafeTensors::open(path)?;
            Ok(Self {
                files: vec![st],
                model_dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            })
        } else {
            // Directory: find all .safetensors files
            let mut files = Vec::new();
            let model_dir = path.to_path_buf();

            // Try model.safetensors first
            let single = path.join("model.safetensors");
            if single.exists() {
                files.push(SafeTensors::open(&single)?);
            } else if let Some(shard_paths) = indexed_shards(path)? {
                // The shards the index names, and no other `.safetensors`
                // file of the directory, such as a head or an adapter.
                for shard_path in &shard_paths {
                    files.push(SafeTensors::open(shard_path)?);
                }
            } else {
                // Sharded: model-00001-of-NNNNN.safetensors
                let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(path)
                    .map_err(|e| Error::ModelError {
                        reason: format!("Failed to read directory: {e}"),
                    })?
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.path())
                    .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
                    .collect();
                shard_paths.sort();

                if shard_paths.is_empty() {
                    return Err(Error::ModelError {
                        reason: format!("No .safetensors files found in {}", path.display()),
                    });
                }

                for shard_path in &shard_paths {
                    files.push(SafeTensors::open(shard_path)?);
                }
            }

            Ok(Self { files, model_dir })
        }
    }

    /// Load one SafeTensors image held in memory, such as a file a browser
    /// fetched. It has no model directory, so [`Self::model_dir`] is empty.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Ok(Self {
            files: vec![SafeTensors::from_bytes(bytes)?],
            model_dir: PathBuf::new(),
        })
    }

    /// Get all tensor names across all files
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for st in &self.files {
            names.extend(st.tensor_names().map(|s| s.to_string()));
        }
        names
    }

    /// Get the underlying SafeTensors for the first file (for detection)
    pub fn first(&self) -> Option<&SafeTensors> {
        self.files.first()
    }

    /// Model directory
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Whether this is a sharded (multi-file) model
    pub fn is_sharded(&self) -> bool {
        self.files.len() > 1
    }

    /// Number of shards (files)
    pub fn num_shards(&self) -> usize {
        self.files.len()
    }

    /// Total size in bytes of all tensors across all shards
    pub fn total_size(&self) -> u64 {
        self.files
            .iter()
            .flat_map(|st| {
                st.tensor_names().map(|name| {
                    st.tensor_info(name)
                        .map(|info| info.size_bytes() as u64)
                        .unwrap_or(0)
                })
            })
            .sum()
    }

    /// Get tensor info by name (searched across all shards)
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        for st in &self.files {
            if let Ok(info) = st.tensor_info(name) {
                return Ok(info);
            }
        }
        Err(Error::ModelError {
            reason: format!("tensor not found in any shard: {name}"),
        })
    }

    /// Read raw tensor bytes by name from whichever shard contains it
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let file_idx = self
            .files
            .iter()
            .position(|st| st.tensor_info(name).is_ok())
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor not found in any shard: {name}"),
            })?;

        self.files[file_idx].read_tensor_bytes(name)
    }

    /// Load a tensor by name from whichever shard contains it
    pub fn load_tensor<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        // Find which file contains the tensor
        let file_idx = self
            .files
            .iter()
            .position(|st| st.tensor_info(name).is_ok())
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor not found in any shard: {name}"),
            })?;

        self.files[file_idx].load_tensor::<R>(name, device)
    }
}

/// The shard index of a sharded checkpoint directory.
const SHARD_INDEX: &str = "model.safetensors.index.json";

/// The shard files [`SHARD_INDEX`] in `dir` names in its `weight_map`,
/// sorted, each once. `None` when `dir` has no index.
///
/// # Errors
///
/// [`Error::ModelError`] naming the index when it does not read or parse,
/// has no `weight_map` object, or maps a tensor to anything but a file
/// name.
fn indexed_shards(dir: &Path) -> Result<Option<Vec<PathBuf>>> {
    let index = dir.join(SHARD_INDEX);
    if !index.is_file() {
        return Ok(None);
    }
    let bad = |reason: String| Error::ModelError {
        reason: format!("{}: {reason}", index.display()),
    };
    let text = std::fs::read_to_string(&index).map_err(|e| bad(format!("cannot read: {e}")))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| bad(format!("does not parse: {e}")))?;
    let map = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| bad("has no weight_map object".into()))?;
    let mut shards = Vec::with_capacity(map.len());
    for (tensor, file) in map {
        let file = file
            .as_str()
            .filter(|f| !f.is_empty())
            .ok_or_else(|| bad(format!("weight_map maps {tensor} to {file}, not a file name")))?;
        shards.push(dir.join(file));
    }
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(bad("weight_map names no shard".into()));
    }
    Ok(Some(shards))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A SafeTensors file holding one F32 tensor `name` of `[1]`.
    fn write_one(path: &Path, name: &str) {
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    /// With an index, only the shards it names load: a head or adapter file
    /// beside them stays out of the model.
    #[test]
    fn an_index_names_the_shards() {
        let dir = tempfile::tempdir().unwrap();
        write_one(&dir.path().join("model-00001-of-00002.safetensors"), "a");
        write_one(&dir.path().join("model-00002-of-00002.safetensors"), "b");
        write_one(&dir.path().join("joint_head.safetensors"), "hidden_norm.weight");
        let globbed = SafeTensorsLoader::open(dir.path()).unwrap();
        assert_eq!(globbed.num_shards(), 3, "no index: every file is a shard");

        let index = r#"{"metadata": {}, "weight_map": {"a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors"}}"#;
        std::fs::write(dir.path().join(SHARD_INDEX), index).unwrap();
        let indexed = SafeTensorsLoader::open(dir.path()).unwrap();
        assert_eq!(indexed.num_shards(), 2);
        let mut names = indexed.tensor_names();
        names.sort();
        assert_eq!(names, ["a", "b"]);

        std::fs::write(dir.path().join(SHARD_INDEX), r#"{"weight_map": {"a": 3}}"#).unwrap();
        let e = SafeTensorsLoader::open(dir.path()).err().unwrap().to_string();
        assert!(e.contains("maps a to 3"), "{e}");
        std::fs::write(dir.path().join(SHARD_INDEX), r#"{"metadata": {}}"#).unwrap();
        let e = SafeTensorsLoader::open(dir.path()).err().unwrap().to_string();
        assert!(e.contains("no weight_map"), "{e}");
    }
}
