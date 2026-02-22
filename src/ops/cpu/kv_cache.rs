//! CPU implementation of KvCacheOps
//!
//! Direct memory copies for cache update and reshape_and_cache.

use crate::error::{Error, Result};
use crate::ops::traits::KvCacheOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

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

        if position + new_len > max_seq_len {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "position {} + new_len {} > max_seq_len {}",
                    position, new_len, max_seq_len
                ),
            });
        }

        let nk = new_k.to_vec::<f32>();
        let nv = new_v.to_vec::<f32>();

        // Write directly into cache memory via raw pointers
        let kc_ptr = k_cache.ptr() as *mut f32;
        let vc_ptr = v_cache.ptr() as *mut f32;

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..new_len {
                    for d in 0..head_dim {
                        let src = ((b * num_heads + h) * new_len + s) * head_dim + d;
                        let dst =
                            ((b * num_heads + h) * max_seq_len + (position + s)) * head_dim + d;
                        unsafe {
                            *kc_ptr.add(dst) = nk[src];
                            *vc_ptr.add(dst) = nv[src];
                        }
                    }
                }
            }
        }

        Ok(())
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

        let k_data = key.to_vec::<f32>();
        let v_data = value.to_vec::<f32>();
        let slots = slot_mapping.to_vec::<i32>();

        let kc_ptr = key_cache.ptr() as *mut f32;
        let vc_ptr = value_cache.ptr() as *mut f32;
        let cache_stride_block = block_size * num_heads * head_dim;

        for (t, &slot_i32) in slots.iter().enumerate().take(num_tokens) {
            let slot = slot_i32 as usize;
            let block_idx = slot / block_size;
            let block_offset = slot % block_size;

            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = (t * num_heads + h) * head_dim + d;
                    let dst = block_idx * cache_stride_block
                        + block_offset * num_heads * head_dim
                        + h * head_dim
                        + d;
                    unsafe {
                        *kc_ptr.add(dst) = k_data[src];
                        *vc_ptr.add(dst) = v_data[src];
                    }
                }
            }
        }

        Ok(())
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
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    fn ones(
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
        shape: &[usize],
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
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
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&[1i32, 5], &[2], &device);

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
}
