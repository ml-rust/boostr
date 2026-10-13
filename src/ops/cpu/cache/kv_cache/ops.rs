//! CPU implementation of KvCacheOps
//!
//! Direct memory copies for cache update and reshape_and_cache.

use super::append_kv_int4;
use super::batched;
use super::block_ops;
use super::reshape_and_cache;
use crate::error::{Error, Result};
use crate::ops::traits::{Int4GroupSize, KvCacheOps};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Checks that every named tensor shares one dtype.
///
/// Returns the shared dtype on success, or an `InvalidArgument` naming the
/// first mismatching tensor, its dtype, the reference tensor, and its dtype.
pub(super) fn check_shared_dtype(tensors: &[(&'static str, &Tensor<CpuRuntime>)]) -> Result<DType> {
    let (first_name, first) = tensors[0];
    let dtype = first.dtype();
    for &(name, t) in &tensors[1..] {
        if t.dtype() != dtype {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "{name} has dtype {:?} but {first_name} has dtype {:?}; all tensors in this call must share one dtype",
                    t.dtype(),
                    dtype
                ),
            });
        }
    }
    Ok(dtype)
}

impl KvCacheOps<CpuRuntime> for CpuClient {
    fn kv_cache_update(
        &self,
        k_cache: &Tensor<CpuRuntime>,
        v_cache: &Tensor<CpuRuntime>,
        new_k: &Tensor<CpuRuntime>,
        new_v: &Tensor<CpuRuntime>,
        position: usize,
    ) -> Result<()> {
        let cache_shape = k_cache.shape();
        let new_shape = new_k.shape();

        if cache_shape.len() != 4 || new_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "shape",
                reason: "expected 4D [B, H, S, D] tensors".into(),
            });
        }

        let batch = cache_shape[0];
        let num_heads = cache_shape[1];
        let max_seq_len = cache_shape[2];
        let head_dim = cache_shape[3];
        let new_len = new_shape[2];

        // Validate that batch, heads, and head_dim are compatible.
        if new_shape[0] != batch || new_shape[1] != num_heads || new_shape[3] != head_dim {
            return Err(Error::InvalidArgument {
                arg: "shape",
                reason: format!(
                    "new_k shape [{},{},{},{}] is incompatible with cache shape [{},{},{},{}]",
                    new_shape[0],
                    new_shape[1],
                    new_len,
                    new_shape[3],
                    batch,
                    num_heads,
                    max_seq_len,
                    head_dim,
                ),
            });
        }

        if position + new_len > max_seq_len {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "position {} + new_len {} > max_seq_len {}",
                    position, new_len, max_seq_len
                ),
            });
        }

        let dtype = check_shared_dtype(&[
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("new_k", new_k),
            ("new_v", new_v),
        ])?;

        // Copy using byte-level operations to support any dtype (F32, BF16, F16, etc.)
        // Use raw pointers directly since both source and dest are CPU memory.
        // Cannot use to_vec::<u8>() because Storage::to_vec uses inner.len (element count)
        // as the number of T values, which gives wrong byte count for non-u8 dtypes.
        let elem_size = dtype.size_in_bytes();
        let nk_ptr = new_k.ptr() as *const u8;
        let nv_ptr = new_v.ptr() as *const u8;
        let kc_ptr = k_cache.ptr() as *mut u8;
        let vc_ptr = v_cache.ptr() as *mut u8;

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..new_len {
                    let src_offset = ((b * num_heads + h) * new_len + s) * head_dim * elem_size;
                    let dst_offset =
                        ((b * num_heads + h) * max_seq_len + (position + s)) * head_dim * elem_size;
                    let row_bytes = head_dim * elem_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            nk_ptr.add(src_offset),
                            kc_ptr.add(dst_offset),
                            row_bytes,
                        );
                        std::ptr::copy_nonoverlapping(
                            nv_ptr.add(src_offset),
                            vc_ptr.add(dst_offset),
                            row_bytes,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn kv_cache_update_batched(
        &self,
        k_caches: &[&Tensor<CpuRuntime>],
        v_caches: &[&Tensor<CpuRuntime>],
        new_ks: &[&Tensor<CpuRuntime>],
        new_vs: &[&Tensor<CpuRuntime>],
        max_seq_len: usize,
        position: usize,
    ) -> Result<()> {
        batched::kv_cache_update_batched(
            self,
            k_caches,
            v_caches,
            new_ks,
            new_vs,
            max_seq_len,
            position,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor<CpuRuntime>,
        value: &Tensor<CpuRuntime>,
        key_cache: &Tensor<CpuRuntime>,
        value_cache: &Tensor<CpuRuntime>,
        slot_mapping: &Tensor<CpuRuntime>,
        block_size: usize,
    ) -> Result<()> {
        reshape_and_cache::reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            block_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn append_kv_int4(
        &self,
        k_cache: &Tensor<CpuRuntime>,
        v_cache: &Tensor<CpuRuntime>,
        k_scales: &Tensor<CpuRuntime>,
        k_zeros: &Tensor<CpuRuntime>,
        v_scales: &Tensor<CpuRuntime>,
        v_zeros: &Tensor<CpuRuntime>,
        new_k: &Tensor<CpuRuntime>,
        new_v: &Tensor<CpuRuntime>,
        position: usize,
        group_size: Int4GroupSize,
    ) -> Result<()> {
        append_kv_int4::append_kv_int4(
            k_cache, v_cache, k_scales, k_zeros, v_scales, v_zeros, new_k, new_v, position,
            group_size,
        )
    }

    fn copy_blocks(
        &self,
        key_cache: &Tensor<CpuRuntime>,
        value_cache: &Tensor<CpuRuntime>,
        block_mapping: &Tensor<CpuRuntime>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<()> {
        block_ops::copy_blocks(
            key_cache,
            value_cache,
            block_mapping,
            num_heads,
            head_dim,
            block_size,
        )
    }

    fn swap_blocks(
        &self,
        src_cache: &Tensor<CpuRuntime>,
        dst_cache: &Tensor<CpuRuntime>,
        block_mapping: &Tensor<CpuRuntime>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<()> {
        block_ops::swap_blocks(
            src_cache,
            dst_cache,
            block_mapping,
            num_heads,
            head_dim,
            block_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{ones, zeros};
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_kv_cache_update_basic() {
        let (client, device) = cpu_setup();
        let k_cache = zeros(&device, &[1, 1, 8, 4]);
        let v_cache = zeros(&device, &[1, 1, 8, 4]);
        let new_k = ones(&device, &[1, 1, 2, 4]);
        let new_v = ones(&device, &[1, 1, 2, 4]);

        client
            .kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, 3)
            .unwrap();

        let kc = k_cache.to_vec::<f32>();
        assert_eq!(kc[0], 0.0); // pos 0
        assert_eq!(kc[3 * 4], 1.0); // pos 3, dim 0
        assert_eq!(kc[4 * 4], 1.0); // pos 4, dim 0
        assert_eq!(kc[5 * 4], 0.0); // pos 5
    }

    #[test]
    fn test_kv_cache_update_overflow() {
        let (client, device) = cpu_setup();
        let k_cache = zeros(&device, &[1, 1, 4, 4]);
        let v_cache = zeros(&device, &[1, 1, 4, 4]);
        let new_k = ones(&device, &[1, 1, 2, 4]);
        let new_v = ones(&device, &[1, 1, 2, 4]);

        let result = client.kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_update_shape_mismatch_is_error() {
        let (client, device) = cpu_setup();
        // cache: [1, 2, 8, 4] but new_k: [1, 1, 2, 4] (heads mismatch)
        let k_cache = zeros(&device, &[1, 2, 8, 4]);
        let v_cache = zeros(&device, &[1, 2, 8, 4]);
        let new_k = ones(&device, &[1, 1, 2, 4]);
        let new_v = ones(&device, &[1, 1, 2, 4]);

        let result = client.kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, 0);
        assert!(
            result.is_err(),
            "head dimension mismatch should be rejected"
        );
    }
}
