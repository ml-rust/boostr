//! Launch geometry and dtype suffix shared by every fused optimizer kernel.
//!
//! Split out of `fused_optimizer.rs`, which keeps the `FusedOptimizerOps`
//! trait wiring; one launcher per optimizer lives in the `fused_optimizer_*`
//! siblings.

use crate::error::{Error, Result};
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;

pub(super) fn launch_cfg(n: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = n.div_ceil(256) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn kernel_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("unsupported dtype {:?} for fused optimizer", dtype),
        }),
    }
}
