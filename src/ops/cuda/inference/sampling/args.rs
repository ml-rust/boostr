//! Host checks of the penalty arguments the CUDA sampling kernels read.
//!
//! The kernels read `token_ids` and `token_counts` through raw pointers for
//! `num_unique` entries. These checks hold both buffers to that length and
//! to the dtypes the kernels read, before a launch.

use crate::error::{Error, Result};
use crate::ops::impl_generic::inference::sampling::validate_contiguous_arg;
use numr::dtype::DType;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

fn one_dim(
    op: &'static str,
    arg: &'static str,
    t: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<usize> {
    validate_contiguous_arg(op, arg, t)?;
    match t.shape() {
        [n] if t.dtype() == dtype => Ok(*n),
        shape => Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {arg} must be a 1-D {dtype:?} tensor, got {:?} of shape {shape:?}",
                t.dtype()
            ),
        }),
    }
}

/// Checks the penalty buffers for `num_unique` entries and returns
/// `num_unique` as the kernels' `int`.
///
/// # Errors
///
/// A non-contiguous buffer, `token_ids` that is not 1-D I64, `token_counts`
/// that is not 1-D I32, a buffer shorter than `num_unique`, or a
/// `num_unique` above `i32::MAX`.
pub(super) fn check_penalty_args(
    op: &'static str,
    token_ids: &Tensor<CudaRuntime>,
    token_counts: &Tensor<CudaRuntime>,
    num_unique: usize,
) -> Result<i32> {
    let ids = one_dim(op, "token_ids", token_ids, DType::I64)?;
    let counts = one_dim(op, "token_counts", token_counts, DType::I32)?;
    if num_unique > ids || num_unique > counts {
        return Err(Error::InvalidArgument {
            arg: "num_unique",
            reason: format!(
                "{op}: num_unique is {num_unique}, but token_ids holds {ids} and token_counts \
                 {counts} entries"
            ),
        });
    }
    i32::try_from(num_unique).map_err(|_| Error::InvalidArgument {
        arg: "num_unique",
        reason: format!(
            "{op}: num_unique {num_unique} exceeds the kernel limit of {}",
            i32::MAX
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cuda_setup;

    #[test]
    fn penalty_buffers_must_cover_num_unique_in_the_kernel_dtypes() {
        let Some(cuda) = cuda_setup() else { return };
        let dev = &cuda.device;
        let ids = Tensor::<CudaRuntime>::from_slice(&[0i64, 3], &[2], dev).unwrap();
        let counts = Tensor::<CudaRuntime>::from_slice(&[1i32, 2], &[2], dev).unwrap();
        assert_eq!(check_penalty_args("op", &ids, &counts, 2).unwrap(), 2);
        assert_eq!(check_penalty_args("op", &ids, &counts, 0).unwrap(), 0);
        let err = check_penalty_args("op", &ids, &counts, 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("num_unique is 3"), "{err}");
        let i32_ids = Tensor::<CudaRuntime>::from_slice(&[0i32, 3], &[2], dev).unwrap();
        assert!(check_penalty_args("op", &i32_ids, &counts, 2).is_err());
        let short = Tensor::<CudaRuntime>::from_slice(&[1i32], &[1], dev).unwrap();
        assert!(check_penalty_args("op", &ids, &short, 2).is_err());
    }
}
