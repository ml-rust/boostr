//! The byte range a saved tensor covers inside its storage.
//!
//! A tensor is a view: `torch.save` writes the WHOLE storage it points into,
//! plus a storage offset, a shape and strides. This module checks the
//! storage size, refuses a strided (non-contiguous) view, and returns the
//! contiguous byte run the view covers.

use super::types::PtTensorMeta;
use crate::error::{Error, Result};
use std::path::Path;

/// The row-major element strides of `shape`, as `torch.Tensor.stride()`
/// reports them for a contiguous tensor.
pub(super) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut step = 1usize;
    for (stride, &size) in strides.iter_mut().zip(shape).rev() {
        *stride = step;
        step = step.saturating_mul(size.max(1));
    }
    strides
}

/// Whether `strides` lay out `shape` contiguously in row-major order.
///
/// Follows `torch.Tensor.is_contiguous`: a size-1 dim can carry any stride,
/// and a tensor with no elements is always contiguous.
pub(super) fn is_row_major(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    if shape.contains(&0) {
        return true;
    }
    let mut expected = 1usize;
    for (&size, &stride) in shape.iter().zip(strides).rev() {
        if size == 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        expected = expected.saturating_mul(size);
    }
    true
}

/// Slice the bytes of the tensor `key` of the file at `path` out of its
/// storage `storage_path`.
///
/// Checks the storage holds exactly the bytes its element count declares,
/// that the strides are row-major, and that the view lies inside the
/// storage. The view starts `meta.storage_offset` elements in.
pub(super) fn tensor_view<'a>(
    path: &Path,
    key: &str,
    meta: &PtTensorMeta,
    storage_bytes: &'a [u8],
    storage_path: &str,
) -> Result<&'a [u8]> {
    let expected_bytes = meta.storage_numel.checked_mul(meta.storage_elem_size);
    if expected_bytes != Some(storage_bytes.len()) {
        return Err(Error::ModelError {
            reason: format!(
                "{}: storage {storage_path} of tensor {key:?} declares {} elements × {} B \
                 but holds {} bytes; the file is truncated or corrupt, so re-save it",
                path.display(),
                meta.storage_numel,
                meta.storage_elem_size,
                storage_bytes.len()
            ),
        });
    }

    if !is_row_major(&meta.shape, &meta.strides) {
        return Err(Error::ModelError {
            reason: format!(
                "{}: tensor {key:?} has shape {:?} and strides {:?}, not the contiguous \
                 strides {:?}; re-save it with `.contiguous()` before `torch.save`",
                path.display(),
                meta.shape,
                meta.strides,
                row_major_strides(&meta.shape)
            ),
        });
    }

    let dtype_bytes = meta.dtype.size_in_bytes();
    let view_numel = meta
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let range = view_numel.and_then(|numel| {
        let start = meta.storage_offset.checked_mul(dtype_bytes)?;
        let len = numel.checked_mul(dtype_bytes)?;
        Some((start, start.checked_add(len)?))
    });
    range
        .and_then(|(start, end)| storage_bytes.get(start..end))
        .ok_or_else(|| Error::ModelError {
            reason: format!(
                "{}: tensor {key:?} of shape {:?} at storage offset {} runs past the end of \
                 storage {storage_path} ({} elements); the file is corrupt, so re-save it",
                path.display(),
                meta.shape,
                meta.storage_offset,
                meta.storage_numel
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_strides_match_torch() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn size_one_dims_take_any_stride() {
        assert!(is_row_major(&[1, 3], &[1, 1]));
        assert!(is_row_major(&[3, 1], &[1, 7]));
        assert!(is_row_major(&[], &[]));
    }

    #[test]
    fn transposed_strides_are_not_row_major() {
        assert!(!is_row_major(&[3, 2], &[1, 3]));
        assert!(!is_row_major(&[3], &[2]));
    }

    #[test]
    fn empty_tensors_are_contiguous() {
        assert!(is_row_major(&[0, 4], &[1, 1]));
    }
}
