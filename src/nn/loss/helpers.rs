use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Flatten targets to [N, 1] for gather operations.
pub(super) fn prepare_targets<R: Runtime<DType = DType>>(
    targets: &Tensor<R>,
    batch_size: usize,
) -> Result<Tensor<R>> {
    targets
        .reshape(&[batch_size])
        .and_then(|t| t.unsqueeze(1))
        .and_then(|t| t.broadcast_to(&[batch_size, 1]))
        .map_err(Error::Numr)
}

/// All dimension indices for a tensor (for full reductions).
pub(super) fn all_dims(ndim: usize) -> Vec<usize> {
    (0..ndim).collect()
}

/// Compute batch size from shape (product of all dims except last).
pub(super) fn batch_size(shape: &[usize]) -> usize {
    if shape.len() <= 1 {
        1
    } else {
        shape[..shape.len() - 1].iter().product()
    }
}
