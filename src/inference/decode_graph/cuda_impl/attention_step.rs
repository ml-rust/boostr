//! Graph-capturable decode attention step: insert this token's K/V into the
//! full-capacity cache at the device-side write position, then attend over
//! the device-side `seq_len_k` span.

use crate::error::Result;
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
/// `sliding_window` is a static config value baked into the captured graph;
/// pass `0` for full attention.
///
/// # Errors
///
/// Propagates errors from `kv_insert` and `decode_attention_graph_fwd`.
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
        kv_cache.capacity(),
        sliding_window,
    )?;

    Ok(attn_out)
}
