//! ZIP archive layout discovery, entry reading, and raw-bytes → tensor.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::io::Read;
use std::path::Path;

pub(super) fn find_archive_layout(
    archive: &zip::ZipArchive<std::fs::File>,
) -> Result<(String, String)> {
    // The archive name is the single top-level directory. Find any entry
    // matching `*/data.pkl` — that tells us both the pkl path and the prefix
    // for storage entries (`{prefix}/data/0`, etc.).
    let names: Vec<String> = archive
        .file_names()
        .filter_map(|name| name.ok().map(|n| n.into_owned()))
        .collect();
    for name in &names {
        if let Some(prefix) = name.strip_suffix("/data.pkl") {
            return Ok((name.clone(), format!("{prefix}/data")));
        }
    }
    // Legacy path: top-level `data.pkl` (very old PyTorch).
    if names.iter().any(|n| n == "data.pkl") {
        return Ok(("data.pkl".to_string(), "data".to_string()));
    }
    Err(Error::ModelError {
        reason: format!("no data.pkl entry found in PyTorch archive; entries: {names:?}"),
    })
}

pub(super) fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::fs::File>,
    name: &str,
) -> Result<Vec<u8>> {
    let mut entry = archive.by_name(name).map_err(|e| Error::ModelError {
        reason: format!("zip entry {name} missing: {e}"),
    })?;
    // The declared size comes from the file, so it is only a hint: cap the
    // up-front allocation and let `read_to_end` grow with the real data.
    const MAX_PREALLOC: u64 = 1 << 28;
    let mut buf = Vec::with_capacity(entry.size().min(MAX_PREALLOC) as usize);
    entry.read_to_end(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("reading zip entry {name}: {e}"),
    })?;
    Ok(buf)
}

/// Decode the little-endian elements of tensor `key` of the file at `path`
/// into a tensor on `device`.
pub(super) fn build_tensor_from_bytes<R: Runtime<DType = DType>>(
    path: &Path,
    key: &str,
    dtype: DType,
    shape: &[usize],
    bytes: &[u8],
    device: &R::Device,
) -> Result<Tensor<R>> {
    match dtype {
        #[cfg(feature = "f16")]
        DType::F16 => {
            let data: Vec<half::f16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device)?)
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            let data: Vec<half::bf16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device)?)
        }
        #[cfg(not(feature = "f16"))]
        DType::F16 | DType::BF16 => Err(Error::ModelError {
            reason: format!(
                "{}: tensor {key:?} is {dtype:?}; build boostr with the `f16` feature to \
                 read F16 and BF16 tensors",
                path.display()
            ),
        }),
        DType::F32 => {
            let data: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device)?)
        }
        DType::F64 => {
            let data: Vec<f64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device)?)
        }
        // Integer buffers: VoxCPM2's `audiovae.pth` carries an I32
        // `sr_bin_boundaries` vector beside its F32 weights, and a converter
        // that copies the whole file has to read it.
        DType::I32 => {
            let data: Vec<i32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device)?)
        }
        DType::I64 => {
            let data: Vec<i64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device)?)
        }
        other => Err(Error::ModelError {
            reason: format!(
                "{}: tensor {key:?} is {other:?}, which the .pt reader does not decode; \
                 convert the file to .safetensors",
                path.display()
            ),
        }),
    }
}
