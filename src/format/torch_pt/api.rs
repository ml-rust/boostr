//! Public loaders: single-tensor reads and the lazy state-dict handle.

use super::io::{build_tensor_from_bytes, find_archive_layout, read_zip_entry};
use super::pickle::parse_pickle;
use super::types::PtTensorMeta;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Load a single tensor from a `.pt` file.
///
/// * If the file's top-level object is a bare tensor, pass `key = None`.
/// * If it's a dict, pass `key = Some("style")` (or whatever key holds your
///   tensor). `load_voice_pt` wraps this with fallback logic for Kokoro.
pub fn load_tensor_pt<R: Runtime<DType = DType>>(
    path: impl AsRef<Path>,
    key: Option<&str>,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::ModelError {
        reason: format!("opening {}: {e}", path.display()),
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| Error::ModelError {
        reason: format!(
            "{} is not a valid PyTorch .pt (ZIP) file: {e}",
            path.display()
        ),
    })?;

    let (pkl_name, data_prefix) = find_archive_layout(&mut archive)?;
    let pkl_bytes = read_zip_entry(&mut archive, &pkl_name)?;
    let contents = parse_pickle(&pkl_bytes)?;

    let wanted_key = key.unwrap_or("");
    let meta = contents.tensors.get(wanted_key).ok_or_else(|| {
        let available: Vec<&str> = contents.tensors.keys().map(String::as_str).collect();
        Error::ModelError {
            reason: format!(
                ".pt file {} has no tensor at key {:?}; available: {:?}",
                path.display(),
                wanted_key,
                available
            ),
        }
    })?;

    let storage_path = format!("{data_prefix}/{}", meta.storage_id);
    let storage_bytes = read_zip_entry(&mut archive, &storage_path)?;

    let view = tensor_view(meta, &storage_bytes, &storage_path)?;
    build_tensor_from_bytes::<R>(meta.dtype, &meta.shape, view, device)
}

/// Convenience wrapper for Kokoro voice files. Tries the bare-tensor form
/// first, then falls back to `{"style": tensor}`.
pub fn load_voice_pt<R: Runtime<DType = DType>>(
    path: impl AsRef<Path>,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let path = path.as_ref();
    match load_tensor_pt::<R>(path, None, device) {
        Ok(t) => Ok(t),
        Err(_) => load_tensor_pt::<R>(path, Some("style"), device),
    }
}

/// Slice the in-storage byte view for `meta`, validating the storage length
/// and that the view lies within bounds.
fn tensor_view<'a>(
    meta: &PtTensorMeta,
    storage_bytes: &'a [u8],
    storage_path: &str,
) -> Result<&'a [u8]> {
    let expected_bytes = meta.storage_numel * meta.storage_elem_size;
    if storage_bytes.len() != expected_bytes {
        return Err(Error::ModelError {
            reason: format!(
                "storage {storage_path} size mismatch: expected {expected_bytes} bytes \
                 ({} elements × {} B), got {}",
                meta.storage_numel,
                meta.storage_elem_size,
                storage_bytes.len()
            ),
        });
    }

    let view_numel: usize = meta.shape.iter().product();
    let dtype_bytes = meta.dtype.size_in_bytes();
    let byte_offset = meta.storage_offset * dtype_bytes;
    let byte_len = view_numel * dtype_bytes;
    if byte_offset + byte_len > storage_bytes.len() {
        return Err(Error::ModelError {
            reason: format!(
                "tensor view exceeds storage: offset={byte_offset} len={byte_len} storage_bytes={}",
                storage_bytes.len()
            ),
        });
    }
    Ok(&storage_bytes[byte_offset..byte_offset + byte_len])
}

/// Whole-state-dict reader: opens a `.pth` once, returns a lazy handle that
/// can hand out tensors by flattened key (e.g. `"bert.embeddings.word_embeddings.weight"`).
///
/// Internally keeps the zip archive open + the parsed pickle metadata. Use
/// `load_tensor(name)` to materialize a specific tensor on demand. The
/// archive is reopened on each `load_tensor` call — cheap for small
/// checkpoints, can be optimized if it shows up on a profile.
pub struct TorchStateDict {
    path: std::path::PathBuf,
    /// Flattened `{dotted.name → tensor metadata}`, plus the data-prefix
    /// string from the archive layout (e.g. `"model/data"`).
    tensors: HashMap<String, PtTensorMeta>,
    data_prefix: String,
}

impl TorchStateDict {
    /// Open a `.pth` / `.pt` state dict and parse its metadata.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error::ModelError {
            reason: format!("opening {}: {e}", path.display()),
        })?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| Error::ModelError {
            reason: format!(
                "{} is not a valid PyTorch .pt (ZIP) file: {e}",
                path.display()
            ),
        })?;
        let (pkl_name, data_prefix) = find_archive_layout(&mut archive)?;
        let pkl_bytes = read_zip_entry(&mut archive, &pkl_name)?;
        let contents = parse_pickle(&pkl_bytes)?;
        Ok(Self {
            path: path.to_path_buf(),
            tensors: contents.tensors,
            data_prefix,
        })
    }

    /// Every flattened tensor key.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Whether a given flattened key is in the state dict.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Materialize a tensor by flattened name. Opens the archive fresh each
    /// call (cheap; the OS page cache keeps hot files warm).
    pub fn load_tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let meta = self.tensors.get(name).ok_or_else(|| Error::ModelError {
            reason: format!(
                "tensor {name:?} not in .pt state dict (have {} tensors)",
                self.tensors.len()
            ),
        })?;

        let file = std::fs::File::open(&self.path).map_err(|e| Error::ModelError {
            reason: format!("reopening {}: {e}", self.path.display()),
        })?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| Error::ModelError {
            reason: format!("reopening archive {}: {e}", self.path.display()),
        })?;
        let storage_path = format!("{}/{}", self.data_prefix, meta.storage_id);
        let storage_bytes = read_zip_entry(&mut archive, &storage_path)?;

        let view = tensor_view(meta, &storage_bytes, &storage_path)?;
        build_tensor_from_bytes::<R>(meta.dtype, &meta.shape, view, device)
    }
}
