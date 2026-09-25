//! CPU implementation of KvCacheOps
//!
//! Direct memory copies for cache update and reshape_and_cache.

use super::append_kv_int4;
use super::batched;
use super::block_ops;
use crate::error::{Error, Result};
use crate::ops::traits::{Int4GroupSize, KvCacheOps};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Checks that every named tensor shares one dtype.
///
/// Returns the shared dtype on success, or an `InvalidArgument` naming the
/// first mismatching tensor, its dtype, the reference tensor, and its dtype.
fn check_shared_dtype(tensors: &[(&'static str, &Tensor<CpuRuntime>)]) -> Result<DType> {
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
        // key/value: [num_tokens, num_heads, head_dim]
        // key_cache/value_cache: [num_blocks, block_size, num_heads, head_dim]
        // slot_mapping: [num_tokens] (i32) — maps each token to a slot index
        let kv_shape = key.shape();
        if kv_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "key",
                reason: "expected 3D [num_tokens, num_heads, head_dim]".into(),
            });
        }

        let num_tokens = kv_shape[0];
        let num_heads = kv_shape[1];
        let head_dim = kv_shape[2];

        let dtype = check_shared_dtype(&[
            ("key", key),
            ("value", value),
            ("key_cache", key_cache),
            ("value_cache", value_cache),
        ])?;

        let slots = slot_mapping.to_vec::<i32>();

        let cache_shape = key_cache.shape();
        if cache_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "key_cache",
                reason: "expected 4D [num_blocks, block_size, num_heads, head_dim]".into(),
            });
        }
        let num_cache_blocks = cache_shape[0];
        let cache_total_elems: usize = cache_shape.iter().product();
        let cache_stride_block = block_size * num_heads * head_dim;

        // Resolve every slot to a (src, dst) element-offset pair up front, with
        // every bounds check the copy loop used to run inline. This keeps the
        // dtype-dispatched copy below free of boostr errors, since its closure
        // is typed `numr::error::Result` (see the comment on it).
        let mut offsets = Vec::with_capacity(num_tokens * num_heads * head_dim);
        for (t, &slot_i32) in slots.iter().enumerate().take(num_tokens) {
            if slot_i32 < 0 {
                return Err(Error::InvalidArgument {
                    arg: "slot_mapping",
                    reason: format!("slot {} at token index {} is negative", slot_i32, t),
                });
            }
            let slot = slot_i32 as usize;
            let block_idx = slot / block_size;
            let block_offset = slot % block_size;

            if block_idx >= num_cache_blocks {
                return Err(Error::InvalidArgument {
                    arg: "slot_mapping",
                    reason: format!(
                        "slot {} maps to block {} which exceeds cache block count {}",
                        slot, block_idx, num_cache_blocks
                    ),
                });
            }

            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = (t * num_heads + h) * head_dim + d;
                    let dst = block_idx * cache_stride_block
                        + block_offset * num_heads * head_dim
                        + h * head_dim
                        + d;
                    if dst >= cache_total_elems {
                        return Err(Error::InvalidArgument {
                            arg: "slot_mapping",
                            reason: format!(
                                "computed cache index {} out of bounds (size {})",
                                dst, cache_total_elems
                            ),
                        });
                    }
                    offsets.push((src, dst));
                }
            }
        }

        // The dtype-erroring arms of `dispatch_dtype!` (unsupported/feature-gated
        // dtype) return numr's own error type, so the copy runs inside a closure
        // and gets mapped to boostr's `Error` at the boundary rather than trying
        // to `return` numr's error type out of this function directly.
        let copied = (|| -> numr::error::Result<()> {
            numr::dispatch_dtype!(dtype, T => {
                let k_data = key.to_vec::<T>();
                let v_data = value.to_vec::<T>();
                let kc_ptr = key_cache.ptr() as *mut T;
                let vc_ptr = value_cache.ptr() as *mut T;

                for &(src, dst) in &offsets {
                    unsafe {
                        *kc_ptr.add(dst) = k_data[src];
                        *vc_ptr.add(dst) = v_data[src];
                    }
                }

                Ok(())
            }, "reshape_and_cache")
        })();
        copied.map_err(Error::Numr)?;

        Ok(())
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
    use super::*;
    use crate::test_utils::cpu_setup;

    fn zeros(
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
        shape: &[usize],
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device).unwrap()
    }

    fn ones(
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
        shape: &[usize],
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device).unwrap()
    }

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
    fn test_reshape_and_cache_basic() {
        let (client, device) = cpu_setup();
        let key = ones(&device, &[2, 1, 4]);
        let value = ones(&device, &[2, 1, 4]);
        let key_cache = zeros(&device, &[4, 2, 1, 4]);
        let value_cache = zeros(&device, &[4, 2, 1, 4]);
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[1i32, 5], &[2], &device).unwrap();

        client
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, 2)
            .unwrap();

        let kc = key_cache.to_vec::<f32>();
        // slot 1 = block 0, offset 1 -> kc[0*8 + 1*4 + 0..4] should be 1.0
        assert_eq!(kc[4], 1.0);
        assert_eq!(kc[5], 1.0);
        // slot 5 = block 2, offset 1 -> kc[2*8 + 1*4 + 0..4] should be 1.0
        assert_eq!(kc[2 * 8 + 4], 1.0);
    }

    #[test]
    fn test_reshape_and_cache_negative_slot_is_error() {
        let (client, device) = cpu_setup();
        let key = ones(&device, &[1, 1, 4]);
        let value = ones(&device, &[1, 1, 4]);
        let key_cache = zeros(&device, &[4, 2, 1, 4]);
        let value_cache = zeros(&device, &[4, 2, 1, 4]);
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[-1i32], &[1], &device).unwrap();

        let result =
            client.reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, 2);
        assert!(result.is_err(), "negative slot should be rejected");
    }

    #[test]
    fn test_reshape_and_cache_out_of_bounds_slot_is_error() {
        let (client, device) = cpu_setup();
        // cache has 2 blocks of size 2 -> valid slots are 0..3
        let key = ones(&device, &[1, 1, 4]);
        let value = ones(&device, &[1, 1, 4]);
        let key_cache = zeros(&device, &[2, 2, 1, 4]);
        let value_cache = zeros(&device, &[2, 2, 1, 4]);
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[4i32], &[1], &device).unwrap();

        let result =
            client.reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, 2);
        assert!(result.is_err(), "slot beyond cache should be rejected");
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

    #[test]
    fn test_reshape_and_cache_dtype_mismatch_is_error() {
        let (client, device) = cpu_setup();
        let key = ones(&device, &[1, 1, 4]); // F32
        let value = ones(&device, &[1, 1, 4]); // F32
        let key_cache = zeros(&device, &[4, 2, 1, 4]); // F32
        let value_cache = zeros(&device, &[4, 2, 1, 4]); // F32
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[1i32], &[1], &device).unwrap();

        #[cfg(feature = "f16")]
        let value_cache = {
            let n: usize = value_cache.shape().iter().product();
            Tensor::<CpuRuntime>::from_slice(
                &vec![half::bf16::from_f32(0.0); n],
                value_cache.shape(),
                &device,
            )
            .unwrap()
        };

        #[cfg(not(feature = "f16"))]
        let value_cache = {
            let n: usize = value_cache.shape().iter().product();
            Tensor::<CpuRuntime>::from_slice(&vec![0i32; n], value_cache.shape(), &device).unwrap()
        };

        let err = client
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, 2)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("F32"), "expected F32 named in error: {msg}");
        assert!(
            msg.contains("key"),
            "expected the reference tensor named in error: {msg}"
        );
        assert!(
            msg.contains("value_cache"),
            "expected the offending tensor named in error: {msg}"
        );
        #[cfg(feature = "f16")]
        assert!(
            msg.contains("BF16"),
            "expected the offending dtype named in error: {msg}"
        );
        #[cfg(not(feature = "f16"))]
        assert!(
            msg.contains("I32"),
            "expected the offending dtype named in error: {msg}"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_reshape_and_cache_bf16_preserves_dtype() {
        use half::bf16;
        use numr::dtype::DType;

        let (client, device) = cpu_setup();
        let vals = [bf16::from_f32(1.0), bf16::from_f32(1.0)];
        let key =
            Tensor::<CpuRuntime>::from_slice(&[vals[0], vals[1]], &[2, 1, 1], &device).unwrap();
        let value =
            Tensor::<CpuRuntime>::from_slice(&[vals[0], vals[1]], &[2, 1, 1], &device).unwrap();
        let key_cache =
            Tensor::<CpuRuntime>::from_slice(&[bf16::from_f32(0.0); 8], &[4, 2, 1, 1], &device)
                .unwrap();
        let value_cache =
            Tensor::<CpuRuntime>::from_slice(&[bf16::from_f32(0.0); 8], &[4, 2, 1, 1], &device)
                .unwrap();
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[1i32, 5], &[2], &device).unwrap();

        client
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, 2)
            .unwrap();

        assert_eq!(key_cache.dtype(), DType::BF16);
        let kc = key_cache.to_vec::<bf16>();
        // slot 1 = block 0, offset 1
        assert_eq!(kc[1], bf16::from_f32(1.0));
        // slot 5 = block 2, offset 1
        assert_eq!(kc[2 * 2 + 1], bf16::from_f32(1.0));
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_reshape_and_cache_f16_preserves_dtype() {
        use half::f16;
        use numr::dtype::DType;

        let (client, device) = cpu_setup();
        let vals = [f16::from_f32(1.0), f16::from_f32(1.0)];
        let key =
            Tensor::<CpuRuntime>::from_slice(&[vals[0], vals[1]], &[2, 1, 1], &device).unwrap();
        let value =
            Tensor::<CpuRuntime>::from_slice(&[vals[0], vals[1]], &[2, 1, 1], &device).unwrap();
        let key_cache =
            Tensor::<CpuRuntime>::from_slice(&[f16::from_f32(0.0); 8], &[4, 2, 1, 1], &device)
                .unwrap();
        let value_cache =
            Tensor::<CpuRuntime>::from_slice(&[f16::from_f32(0.0); 8], &[4, 2, 1, 1], &device)
                .unwrap();
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[1i32, 5], &[2], &device).unwrap();

        client
            .reshape_and_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, 2)
            .unwrap();

        assert_eq!(key_cache.dtype(), DType::F16);
        let kc = key_cache.to_vec::<f16>();
        assert_eq!(kc[1], f16::from_f32(1.0));
        assert_eq!(kc[2 * 2 + 1], f16::from_f32(1.0));
    }
}
