//! Preconditions for [`MiniCpm4Attention::forward_cached`](super::MiniCpm4Attention::forward_cached).
//!
//! Split out of `attention.rs` to keep that file under the crate's 500-line
//! limit for model files. Each function here turns an invariant the cached
//! path depends on into a loud error rather than a silently wrong tensor.

use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

use crate::error::{Error, Result};
use crate::inference::kv_cache::KvCache;

/// Reject a cache that would GROW while writing `seq` more slots.
///
/// The in-place `update_fused` write and the flash read of `k_cache_raw` both
/// assume one buffer for the whole sequence. A grow reallocates it and copies
/// every written slot across — the per-step full-buffer copy this path exists
/// to avoid. VoxCPM2 never hits it (`new_kv_cache` passes
/// `initial_capacity == max_seq_len`); this makes that invariant loud.
pub(super) fn require_preallocated_cache<R: Runtime<DType = DType>>(
    kv_cache: &KvCache<R>,
    seq: usize,
) -> Result<()>
where
    R::Client: IndexingOps<R>,
{
    let (written, capacity) = (kv_cache.seq_len(), kv_cache.capacity());
    if written + seq > capacity {
        return Err(Error::InferenceError {
            reason: format!(
                "MiniCPM4 cached attention needs a KV cache preallocated to its full \
                 width: writing {seq} slots at position {written} needs capacity {}, \
                 got {capacity}. Build the cache with initial_capacity == max_seq_len, \
                 as MiniCpm4Model::new_kv_cache does.",
                written + seq,
            ),
        });
    }
    Ok(())
}

/// The error for a `None` RoPE table on a block that rotates.
///
/// Shared by both paths so neither can degrade into an unrotated forward that
/// stays shape-valid while computing a different model.
pub(super) fn missing_rope() -> Error {
    Error::InvalidArgument {
        arg: "rope",
        reason: "expected Some(RoPE) for a MiniCPM4 block with no_rope unset, got None; \
                 only a no_rope (NoPE) block runs without a rotary table"
            .to_string(),
    }
}
