//! `kv_cache_update_batched` CPU reference implementation.
//!
//! Split out of `kv_cache.rs` to keep it under the `cpu/*.rs` 400-line limit.
//! Plain per-layer loop calling the same byte-copy logic as
//! `kv_cache_update`, kept separate for clarity over speed — this is the
//! oracle the CUDA batched kernel is checked against, not a hot path.

use crate::error::{Error, Result};
use crate::ops::traits::KvCacheOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

pub(super) fn kv_cache_update_batched(
    client: &CpuClient,
    k_caches: &[&Tensor<CpuRuntime>],
    v_caches: &[&Tensor<CpuRuntime>],
    new_ks: &[&Tensor<CpuRuntime>],
    new_vs: &[&Tensor<CpuRuntime>],
    max_seq_len: usize,
    position: usize,
) -> Result<()> {
    let num_layers = k_caches.len();
    if num_layers == 0
        || v_caches.len() != num_layers
        || new_ks.len() != num_layers
        || new_vs.len() != num_layers
    {
        return Err(Error::InvalidArgument {
            arg: "k_caches",
            reason: format!(
                "kv_cache_update_batched requires non-empty, equal-length slices, got {} k_caches, {} v_caches, {} new_ks, {} new_vs",
                num_layers,
                v_caches.len(),
                new_ks.len(),
                new_vs.len(),
            ),
        });
    }

    let cache_shape = k_caches[0].shape().to_vec();
    let new_shape = new_ks[0].shape().to_vec();
    if cache_shape.len() != 4 || new_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k_caches",
            reason: "expected 4D [B, H, S, D] tensors".into(),
        });
    }
    if cache_shape[2] != max_seq_len {
        return Err(Error::InvalidArgument {
            arg: "max_seq_len",
            reason: format!(
                "cache seq dim {} does not match max_seq_len {max_seq_len}",
                cache_shape[2]
            ),
        });
    }
    let new_len = new_shape[2];
    if position + new_len > max_seq_len {
        return Err(Error::InvalidArgument {
            arg: "position",
            reason: format!("position {position} + new_len {new_len} > max_seq_len {max_seq_len}"),
        });
    }

    for layer in 0..num_layers {
        if k_caches[layer].shape() != cache_shape.as_slice()
            || v_caches[layer].shape() != cache_shape.as_slice()
        {
            return Err(Error::InvalidArgument {
                arg: "k_caches",
                reason: format!(
                    "layer {layer}: cache shape {:?} does not match layer 0's {:?} — every layer must share one shape",
                    k_caches[layer].shape(),
                    cache_shape
                ),
            });
        }
        if new_ks[layer].shape() != new_shape.as_slice()
            || new_vs[layer].shape() != new_shape.as_slice()
        {
            return Err(Error::InvalidArgument {
                arg: "new_ks",
                reason: format!(
                    "layer {layer}: new tensor shape {:?} does not match layer 0's {:?} — every layer must share one shape",
                    new_ks[layer].shape(),
                    new_shape
                ),
            });
        }

        client.kv_cache_update(
            k_caches[layer],
            v_caches[layer],
            new_ks[layer],
            new_vs[layer],
            position,
        )?;
    }

    Ok(())
}
