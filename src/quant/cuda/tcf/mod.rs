//! CUDA support for TCF native quantized weights.
//!
//! [`layout`] turns `tcf-core`'s own `QuantLayout` into the plane offsets a
//! kernel takes; [`launch`] drives the three kernels in `kernels/tcf.cu`.

mod launch;
mod layout;

pub(super) use launch::{MatmulShape, launch_dequant, launch_gemm, launch_gemv};
