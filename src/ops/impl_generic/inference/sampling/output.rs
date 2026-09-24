//! Shared device write for a sampled token, used by `sample_token` and
//! `logits_to_token`'s generic implementation.

use crate::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Writes `token_id` into `output`'s device buffer, byte for byte.
///
/// `output` must be a `[1]` I64 tensor. `op` names the calling method in
/// any error.
///
/// # Errors
///
/// Propagates a device copy error, wrapped with `op`'s name.
pub(super) fn write_token_id<R: Runtime>(
    op: &'static str,
    token_id: i64,
    output: &Tensor<R>,
) -> Result<()> {
    let bytes = token_id.to_ne_bytes();
    R::copy_to_device(&bytes, output.ptr(), output.device()).map_err(|e| {
        Error::Numr(numr::error::Error::Internal(format!(
            "{op}: writing the sampled token into scratch.output failed: {e}"
        )))
    })
}
