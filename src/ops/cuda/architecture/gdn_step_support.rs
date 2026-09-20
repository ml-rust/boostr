//! Launch-side pieces `gdn_step` and `gdn_step_from_conv` share: dtype
//! checks, the column-chunk launch grid and the launch-failure error.

use crate::error::{Error, Result};
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Threads per block; matches `GDN_STEP_BLOCK` in the kernel.
pub const BLOCK: u32 = 256;

/// Every tensor must be F32, or `Error::InvalidArgument` naming the first
/// mismatch.
pub fn require_f32(fn_name: &str, tensors: &[(&'static str, &Tensor<CudaRuntime>)]) -> Result<()> {
    for &(name, t) in tensors {
        if t.dtype() != DType::F32 {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "{fn_name} takes F32, got {:?} for shape {:?}",
                    t.dtype(),
                    t.shape()
                ),
            });
        }
    }
    Ok(())
}

/// Grid of column chunks (x) by (batch * head) (y), one thread per column.
pub fn launch_config(s_v: usize, num_bh: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((s_v as u32).div_ceil(BLOCK), num_bh as u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Wraps a kernel launch failure with the kernel name and state shape.
pub fn launch_error(name: &str, state_shape: &[usize], e: impl std::fmt::Debug) -> Error {
    Error::KernelError {
        reason: format!("{name} launch failed for state shape {state_shape:?}: {e:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_config_grid_covers_columns() {
        let cfg = launch_config(300, 6);
        assert_eq!(cfg.grid_dim, (2, 6, 1));
        assert_eq!(cfg.block_dim, (BLOCK, 1, 1));
    }
}
