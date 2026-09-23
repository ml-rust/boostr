//! Graph-capturable decode attention step: insert this token's K/V into the
//! full-capacity cache at the device-side write position, then attend over
//! each row's own device-side key range.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::inference::decode_graph::DeviceScalars;
use crate::inference::lanes::LaneKvStep;
use crate::ops::cuda::attention::flash::impl_ops::decode_attention_graph_fwd;
use crate::ops::cuda::attention::kv_insert::kv_insert;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Insert `k`/`v` into `kv_cache` at `device_scalars.write_pos_ptr()`, then run
/// decode attention for `q` over each row's live slots of the cache.
///
/// `device_scalars` carries one position per lane, so `k` must hold exactly
/// as many rows as it has lanes: `kv_insert` reads `write_pos[b]` for row
/// `b`, and a shorter buffer would be read past its end.
///
/// `lanes` carries the per-row key range. With it, row `b` attends over
/// `[kv_start[b], kv_end[b])`, so one captured launch serves lanes of
/// different live lengths. Without it every row shares the scalars' bound.
///
/// `sliding_window` is a static config value baked into the captured graph;
/// pass `0` for full attention.
///
/// # Errors
///
/// [`Error::InferenceError`] when `k`'s batch is not the scalars' lane
/// count, or a lane bound is not `[batch]` I32; errors from `kv_insert` and
/// `decode_attention_graph_fwd` propagate.
#[allow(clippy::too_many_arguments)]
pub fn insert_and_decode_attention(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    kv_cache: &KvCache<CudaRuntime>,
    device_scalars: &DeviceScalars,
    lanes: Option<&LaneKvStep<'_, CudaRuntime>>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
) -> Result<Tensor<CudaRuntime>> {
    let batch = k.shape().first().copied().unwrap_or(0);
    if batch != device_scalars.batch() {
        return Err(Error::InferenceError {
            reason: format!(
                "decode step has {batch} rows but the decode scalars hold {} lanes; allocate \
                 the scalars with one lane per row of this launch",
                device_scalars.batch()
            ),
        });
    }

    // Device addresses of the per-row range, or 0 for a launch whose rows
    // all share the scalars' bound.
    let (kv_start_ptr, kv_end_ptr) = match lanes {
        Some(lanes) => {
            for (name, bound) in [("kv_start", lanes.kv_start), ("kv_end", lanes.kv_end)] {
                if bound.shape() != [batch].as_slice() || bound.dtype() != numr::dtype::DType::I32 {
                    return Err(Error::InferenceError {
                        reason: format!(
                            "decode step has {batch} rows but {name} is {:?} {:?}; allocate the \
                             lane table with one lane per row of this launch",
                            bound.shape(),
                            bound.dtype()
                        ),
                    });
                }
            }
            (lanes.kv_start.ptr(), lanes.kv_end.ptr())
        }
        None => (0, 0),
    };

    kv_insert(
        client,
        k,
        v,
        kv_cache.k_cache_raw(),
        kv_cache.v_cache_raw(),
        device_scalars.write_pos_ptr(),
    )?;

    let (attn_out, _lse) = decode_attention_graph_fwd(
        client,
        q,
        kv_cache.k_cache_raw(),
        kv_cache.v_cache_raw(),
        num_heads,
        num_kv_heads,
        head_dim,
        device_scalars.seq_len_k_ptr(),
        kv_start_ptr,
        kv_end_ptr,
        kv_cache.capacity(),
        sliding_window,
    )?;

    Ok(attn_out)
}
