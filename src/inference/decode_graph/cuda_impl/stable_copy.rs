//! Graph-capturable device-to-device copy into a pre-allocated buffer.
//!
//! A value the next replay or the host reads must live at an address
//! allocated before capture. Producing it in place is not always possible:
//! `argmax`, `gdn_step` and `causal_conv1d` each return a fresh tensor. This
//! copy records one `cuMemcpyDtoDAsync` node whose source CUDA re-patches
//! per replay and whose destination address never changes.

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Copy every byte of `src` into `dst` on the client's compute stream.
///
/// Valid inside and outside a capture. Inside a capture `src` can be
/// graph-managed; `dst` must be allocated before capture.
///
/// # Errors
///
/// `InvalidArgument` when the dtypes differ, the element counts differ, or
/// either tensor is not contiguous. `Backend` when the driver call fails.
pub fn copy_into_stable(
    client: &CudaClient,
    src: &Tensor<CudaRuntime>,
    dst: &Tensor<CudaRuntime>,
) -> numr::error::Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(numr::error::Error::InvalidArgument {
            arg: "dst",
            reason: format!(
                "copy_into_stable: dtype {:?} != src dtype {:?}",
                dst.dtype(),
                src.dtype()
            ),
        });
    }
    if src.numel() != dst.numel() {
        return Err(numr::error::Error::InvalidArgument {
            arg: "dst",
            reason: format!(
                "copy_into_stable: {} elements != src {} elements (shapes {:?} vs {:?})",
                dst.numel(),
                src.numel(),
                dst.shape(),
                src.shape()
            ),
        });
    }
    if !src.is_contiguous() || !dst.is_contiguous() {
        return Err(numr::error::Error::InvalidArgument {
            arg: "src",
            reason: "copy_into_stable: both tensors must be contiguous".into(),
        });
    }
    let bytes = src.numel() * src.dtype().size_in_bytes();
    if bytes == 0 {
        return Ok(());
    }
    unsafe {
        let result =
            sys::cuMemcpyDtoDAsync_v2(dst.ptr(), src.ptr(), bytes, client.stream().cu_stream());
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(numr::error::Error::Backend(format!(
                "copy_into_stable cuMemcpyDtoDAsync_v2 failed: {:?}",
                result
            )));
        }
    }
    Ok(())
}
