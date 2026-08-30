//! Shared helper functions for CPU quant_matmul implementations.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Validate input is F32 and extract (M, K) from shape.
pub(super) fn validate_input(input: &Tensor<CpuRuntime>) -> Result<(usize, usize)> {
    if input.dtype() != DType::F32 {
        return Err(Error::QuantError {
            reason: format!("input must be F32, got {:?}", input.dtype()),
        });
    }
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(Error::QuantError {
            reason: format!("input must be at least 2D, got {:?}", shape),
        });
    }
    let k = shape[shape.len() - 1];
    let m: usize = shape.iter().product::<usize>() / k;
    Ok((m, k))
}

/// Build output shape: replace last dim with n.
pub(super) fn output_shape(input_shape: &[usize], n: usize) -> Vec<usize> {
    let mut s = input_shape[..input_shape.len() - 1].to_vec();
    s.push(n);
    s
}

/// The `m * k` activation values a tensor VIEW actually holds.
///
/// `Storage::as_host_slice` hands back the WHOLE allocation, never the view's
/// window. A row narrowed out of a `[1, T, K]` batch is contiguous and holds
/// `m * k` values, but its storage still holds `T * K` of them, so reading
/// from the start of that slice multiplies the wrong rows — silently,
/// whenever the view begins past element zero. Taking the window here is what
/// stops that, and it is why every kernel below is handed a slice rather than
/// a tensor.
///
/// # Errors
/// [`Error::QuantError`] when the tensor is not contiguous (its elements are
/// then not one run of memory at all), or when the window falls outside its
/// storage.
pub(super) fn activation_window(
    activation: &Tensor<CpuRuntime>,
    m: usize,
    k: usize,
) -> Result<&[f32]> {
    if !activation.is_contiguous() {
        return Err(Error::QuantError {
            reason: "activation must be contiguous to be read as a flat slice".into(),
        });
    }
    let start = activation.offset();
    let end = start
        .checked_add(m.saturating_mul(k))
        .ok_or_else(|| Error::QuantError {
            reason: format!("activation window {start}+{m}x{k} overflows"),
        })?;
    // SAFETY: CpuRuntime stores data as host pointers.
    let all = unsafe { activation.storage().as_host_slice::<f32>() };
    all.get(start..end).ok_or_else(|| Error::QuantError {
        reason: format!(
            "activation window [{start}, {end}) falls outside its {} value storage",
            all.len()
        ),
    })
}
