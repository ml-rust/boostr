//! Graph-capturable device-to-device copy into a pre-allocated buffer.
//!
//! A value the next replay or the host reads must live at an address
//! allocated before capture. Producing it in place is not always possible:
//! `argmax`, `gdn_step` and `causal_conv1d` each return a fresh tensor. This
//! copy records one memcpy node whose source CUDA re-patches per replay and
//! whose destination address never changes.

use numr::ops::BinaryOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Copy every element of `src` into `dst` on the client's compute stream.
///
/// `src` and `dst` need the same dtype and element count; the shapes can
/// differ (`argmax` yields `[1, 1]` for a `[1]` token buffer). Valid inside
/// and outside a capture. Inside a capture `src` can be graph-managed;
/// `dst` must be allocated before capture.
///
/// # Errors
///
/// `ShapeMismatch` when the element counts differ, `DTypeMismatch` when the
/// dtypes differ, `Backend` when `dst` is not contiguous or the copy fails.
pub fn copy_into_stable(
    client: &CudaClient,
    src: &Tensor<CudaRuntime>,
    dst: &Tensor<CudaRuntime>,
) -> numr::error::Result<()> {
    let src = if src.shape() == dst.shape() {
        src.clone()
    } else {
        src.contiguous()?.reshape(dst.shape())?
    };
    client.copy_into(dst, &src)
}
