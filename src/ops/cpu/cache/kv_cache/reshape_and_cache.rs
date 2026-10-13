//! CPU `reshape_and_cache`: scatters `[num_tokens, num_heads, head_dim]`
//! keys and values into the paged `[num_blocks, block_size, num_heads,
//! head_dim]` caches through the slot mapping.

use super::ops::check_shared_dtype;
use crate::error::{Error, Result};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

pub(super) fn reshape_and_cache(
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

#[cfg(test)]
mod tests {
    use super::super::test_support::{ones, zeros};
    use crate::ops::traits::KvCacheOps;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
