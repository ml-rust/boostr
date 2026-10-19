//! Value readback and comparison helpers for unit tests.

use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// The bit patterns of `values`, for exact equality checks.
pub(crate) fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The largest `|a[i] - b[i]|`. Panics when the lengths differ.
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// F64 form of [`max_abs_diff`].
pub(crate) fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// [`max_abs_diff`] over two F32 CPU tensors.
pub(crate) fn tensor_max_abs_diff(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
    max_abs_diff(&a.to_vec::<f32>(), &b.to_vec::<f32>())
}

/// The values of an F64 or F32 CPU tensor as F64. F32 widens exactly.
pub(crate) fn to_f64(t: &Tensor<CpuRuntime>) -> Vec<f64> {
    let t = t.contiguous().unwrap();
    match t.dtype() {
        DType::F64 => t.to_vec::<f64>(),
        _ => t.to_vec::<f32>().iter().map(|v| f64::from(*v)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_abs_diff_picks_the_largest_gap() {
        assert_eq!(max_abs_diff(&[1.0, 2.0, 3.0], &[1.0, 2.5, 2.0]), 1.0);
        assert_eq!(max_abs_diff_f64(&[0.0], &[-0.25]), 0.25);
    }

    #[test]
    fn bits_tells_signed_zeros_apart() {
        assert_ne!(bits(&[0.0]), bits(&[-0.0]));
    }
}
