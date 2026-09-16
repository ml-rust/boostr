//! Graph-capturable argmax: writes into a pre-allocated stable buffer so the
//! result survives outside the CUDA graph that produced it.

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Argmax on `logits` (graph-internal tensor) and write result into `out` (stable).
///
/// This function MUST be called inside a `Runtime::capture_graph_into()` closure.
/// The source (`logits`) has a graph-managed address — CUDA internally patches it
/// on each replay.  The destination (`out`) is pre-allocated before capture and
/// has a stable address that the caller can read after each graph launch.
///
/// `logits` shape: `[1, 1, vocab_size]`
/// `out` shape: `[1]` i64
pub fn argmax_to_buf(
    client: &CudaClient,
    logits: &Tensor<CudaRuntime>,
    out: &Tensor<CudaRuntime>,
) -> numr::error::Result<()> {
    use numr::ops::traits::IndexingOps;

    // Argmax along last dim: [1, 1, vocab] → [1, 1] i64
    // Allocated inside the graph (graph-managed address — unstable from CPU)
    let last_dim = logits.shape().len() - 1;
    let token_ids = client.argmax(logits, last_dim, false)?;

    // cuMemcpyAsync: from graph-internal token_ids to pre-allocated `out`.
    // CUDA records a MemCpy node; on each graph replay it patches the source
    // address to the actual execution-time allocation of `token_ids`.
    let bytes = std::mem::size_of::<i64>();
    unsafe {
        let result = sys::cuMemcpyAsync(
            out.ptr(),
            token_ids.ptr(),
            bytes,
            client.stream().cu_stream(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(numr::error::Error::Backend(format!(
                "argmax_to_buf cuMemcpyAsync failed: {:?}",
                result
            )));
        }
    }
    Ok(())
}

/// Argmax per row of `logits` and write the `batch_size` results into `out`.
///
/// Like `argmax_to_buf` but handles a batched logits tensor of shape
/// `[batch_size, 1, vocab_size]`, writing one i64 argmax per row into
/// `out` (pre-allocated, shape `[batch_size]`).
///
/// This function MUST be called inside a `Runtime::capture_graph_into()` closure.
/// The `logits` tensor has a graph-managed address; `out` must be pre-allocated
/// before capture so its address is stable across replays.
pub fn batch_argmax_to_buf(
    client: &CudaClient,
    logits: &Tensor<CudaRuntime>,
    out: &Tensor<CudaRuntime>,
    batch_size: usize,
) -> numr::error::Result<()> {
    use numr::ops::traits::IndexingOps;

    // Argmax along vocab dim: [B, 1, vocab] → [B, 1] → [B]
    let last_dim = logits.shape().len() - 1;
    let token_ids = client.argmax(logits, last_dim, false)?;
    // token_ids: [batch_size, 1] i64

    // Copy all B argmax results to the pre-allocated stable output buffer.
    // CUDA records a MemCpy node; on replay it patches the source address.
    let bytes = batch_size * std::mem::size_of::<i64>();
    unsafe {
        let result = sys::cuMemcpyAsync(
            out.ptr(),
            token_ids.ptr(),
            bytes,
            client.stream().cu_stream(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(numr::error::Error::Backend(format!(
                "batch_argmax_to_buf cuMemcpyAsync failed: {:?}",
                result
            )));
        }
    }
    Ok(())
}
