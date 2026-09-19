//! Graph-capturable argmax: writes into a pre-allocated stable buffer so the
//! result survives outside the CUDA graph that produced it.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::stable_copy::copy_into_stable;

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

    // Captured memcpy node from graph-internal `token_ids` to pre-allocated `out`.
    copy_into_stable(client, &token_ids, out)
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
    if token_ids.numel() != batch_size {
        return Err(numr::error::Error::InvalidArgument {
            arg: "batch_size",
            reason: format!(
                "batch_argmax_to_buf: argmax produced {} rows, batch_size is {batch_size}",
                token_ids.numel()
            ),
        });
    }
    copy_into_stable(client, &token_ids, out)
}
