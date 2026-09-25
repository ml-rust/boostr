//! Shared device-side seed write for the first decode-graph input token.

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};

/// Write `token` into element 0 of `buf` (a stable i64 tensor).
///
/// No host pointer is involved, so there is no stack-lifetime hazard. Call once
/// before the decode loop starts to seed the first input token.
pub(super) fn seed_i64(client: &CudaClient, buf: &Tensor<CudaRuntime>, token: i64) -> Result<()> {
    seed_i64_at(client, buf, 0, token)
}

/// Write `token` into element `index` of `buf` (a stable i64 tensor) with two
/// stream-ordered `cuMemsetD32Async` calls: the low word at the element's
/// address, the high word four bytes past it.
///
/// # Errors
///
/// [`Error::InferenceError`] when `index` is outside `buf`, or a memset
/// fails.
pub(super) fn seed_i64_at(
    client: &CudaClient,
    buf: &Tensor<CudaRuntime>,
    index: usize,
    token: i64,
) -> Result<()> {
    if index >= buf.numel() {
        return Err(Error::InferenceError {
            reason: format!(
                "seed_i64_at: element {index} is outside a buffer of {}; seed an element the \
                 buffer holds",
                buf.numel()
            ),
        });
    }
    let base = buf.ptr() + (index * std::mem::size_of::<i64>()) as u64;
    let lo = (token as u64 & 0xFFFF_FFFF) as u32;
    let hi = ((token as u64) >> 32) as u32;
    // Both memsets are driver enqueues made outside numr's launch path, so
    // they hold the capture permit themselves for the pair.
    let _permit = client.stream().enqueue_permit();
    let stream = client.stream().raw().cu_stream();
    unsafe {
        let result = sys::cuMemsetD32Async(base, lo, 1, stream);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::InferenceError {
                reason: format!("seed_i64_at cuMemsetD32Async lo failed: {:?}", result),
            });
        }
        let result = sys::cuMemsetD32Async(base + 4, hi, 1, stream);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::InferenceError {
                reason: format!("seed_i64_at cuMemsetD32Async hi failed: {:?}", result),
            });
        }
    }
    Ok(())
}
