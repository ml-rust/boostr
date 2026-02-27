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
