//! Graph-capturable decode attention step: insert this token's K/V into the
//! full-capacity cache at the device-side write position, then attend over
//! the device-side `seq_len_k` span.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::inference::decode_graph::DeviceScalars;
use crate::ops::cuda::attention::flash::impl_ops::decode_attention_graph_fwd;
use crate::ops::cuda::attention::kv_insert::kv_insert;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Insert `k`/`v` into `kv_cache` at `device_scalars.write_pos_ptr()`, then run
/// decode attention for `q` over the first `*device_scalars.seq_len_k_ptr()`
/// slots of the cache.
///
/// `device_scalars` carries one position per lane, so `k` must hold exactly
/// as many rows as it has lanes: `kv_insert` reads `write_pos[b]` for row
/// `b`, and a shorter buffer would be read past its end.
///
/// `sliding_window` is a static config value baked into the captured graph;
/// pass `0` for full attention.
///
/// # Errors
///
/// [`Error::InferenceError`] when `k`'s batch is not the scalars' lane
/// count; errors from `kv_insert` and `decode_attention_graph_fwd`
/// propagate.
#[allow(clippy::too_many_arguments)]
pub fn insert_and_decode_attention(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    kv_cache: &KvCache<CudaRuntime>,
    device_scalars: &DeviceScalars,
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
        // No per-row start or end: every row of this launch shares one length.
        0,
        0,
        kv_cache.capacity(),
        sliding_window,
    )?;

    Ok(attn_out)
}
