//! CUDA support for TCF native quantized weights.
//!
//! `launch` drives the three kernels in `kernels/tcf.cu`. The plane offsets
//! they take come from `quant::tcf::TcfPlanes`, which reads them off
//! `tcf-core`'s own `QuantLayout`, so the shader-side code holds no plane
//! order and no plane size.

mod gemv_dp4a;
mod launch;

pub(super) use gemv_dp4a::{DP4A_GEMV_MAX_TOKENS, launch_gemv_dp4a, supports_dp4a_gemv};
pub(super) use launch::{MatmulShape, launch_dequant, launch_gemm, launch_gemv};
