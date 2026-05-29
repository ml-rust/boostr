//! ZIP archive layout discovery, entry reading, and raw-bytes → tensor.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::io::Read;

pub(super) fn find_archive_layout(
    archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<(String, String)> {
    // The archive name is the single top-level directory. Find any entry
    // matching `*/data.pkl` — that tells us both the pkl path and the prefix
    // for storage entries (`{prefix}/data/0`, etc.).
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
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
        reason: format!(
            "no data.pkl entry found in PyTorch archive; entries: {:?}",
            &names
        ),
    })
}

pub(super) fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::fs::File>,
    name: &str,
) -> Result<Vec<u8>> {
    let mut entry = archive.by_name(name).map_err(|e| Error::ModelError {
        reason: format!("zip entry {name} missing: {e}"),
    })?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("reading zip entry {name}: {e}"),
    })?;
    Ok(buf)
}

pub(super) fn build_tensor_from_bytes<R: Runtime<DType = DType>>(
    dtype: DType,
    shape: &[usize],
    bytes: &[u8],
    device: &R::Device,
) -> Result<Tensor<R>> {
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device))
        }
        DType::F64 => {
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Ok(Tensor::<R>::from_slice(&data, shape, device))
        }
        other => Err(Error::ModelError {
            reason: format!(
                "reading {other:?} tensors from .pt is not yet supported — convert to .safetensors"
            ),
        }),
    }
}
