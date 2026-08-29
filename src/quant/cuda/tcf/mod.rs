//! CUDA support for TCF native quantized weights.
//!
//! `launch` drives the three kernels in `kernels/tcf.cu`. The plane offsets
//! they take come from `quant::tcf::TcfPlanes`, which reads them off
//! `tcf-core`'s own `QuantLayout`, so the shader-side code holds no plane
//! order and no plane size.

mod launch;

pub(super) use launch::{MatmulShape, launch_dequant, launch_gemm, launch_gemv};
