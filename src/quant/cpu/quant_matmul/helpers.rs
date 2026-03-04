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
