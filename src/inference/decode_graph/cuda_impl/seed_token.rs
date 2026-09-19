//! Shared device-side seed write for the first decode-graph input token.

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};

/// Write `token` into `buf` (a stable `[1]` i64 tensor) with two stream-ordered
/// `cuMemsetD32Async` calls (low word at `buf.ptr()`, high word at `buf.ptr() + 4`).
///
/// No host pointer is involved, so there is no stack-lifetime hazard. Call once
/// before the decode loop starts to seed the first input token.
pub(super) fn seed_i64(client: &CudaClient, buf: &Tensor<CudaRuntime>, token: i64) -> Result<()> {
    let lo = (token as u64 & 0xFFFF_FFFF) as u32;
    let hi = ((token as u64) >> 32) as u32;
    let stream = client.stream().cu_stream();
    unsafe {
        let result = sys::cuMemsetD32Async(buf.ptr(), lo, 1, stream);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::InferenceError {
                reason: format!("seed_i64 cuMemsetD32Async lo failed: {:?}", result),
            });
        }
        let result = sys::cuMemsetD32Async(buf.ptr() + 4, hi, 1, stream);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::InferenceError {
                reason: format!("seed_i64 cuMemsetD32Async hi failed: {:?}", result),
            });
        }
    }
    Ok(())
}
