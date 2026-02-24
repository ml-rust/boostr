//! Backend parity tests for KvCacheOps.

use super::helpers::*;
use boostr::ops::traits::cache::kv_cache::KvCacheOps;

#[test]
fn test_kv_cache_update_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 16, 32);
    let new_len = 4;
    let position = 3;

    // Zero-init caches
    let zeros = vec![0.0f32; b * kv_heads * max_seq * d];
    let k_cache = numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device);
    let v_cache = numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device);
    let new_k = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);
    let new_v = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);

    cpu_client
        .kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, position)
        .unwrap();
    let cpu_k = k_cache.to_vec::<f32>();
    let cpu_v = v_cache.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k_c = Tensor::from_slice(
            &vec![0.0f32; b * kv_heads * max_seq * d],
            &[b, kv_heads, max_seq, d],
            &cuda_device,
        );
        let v_c = Tensor::from_slice(
            &vec![0.0f32; b * kv_heads * max_seq * d],
            &[b, kv_heads, max_seq, d],
            &cuda_device,
        );
        let nk = Tensor::from_slice(
            &new_k.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        );
        let nv = Tensor::from_slice(
            &new_v.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        );
        cuda_client
            .kv_cache_update(&k_c, &v_c, &nk, &nv, position)
            .unwrap();
        assert_parity_f32(
            &k_c.to_vec::<f32>(),
            &cpu_k,
            "kv_cache_update K CUDA vs CPU",
        );
        assert_parity_f32(
            &v_c.to_vec::<f32>(),
            &cpu_v,
            "kv_cache_update V CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k_w = Tensor::from_slice(
            &vec![0.0f32; b * kv_heads * max_seq * d],
            &[b, kv_heads, max_seq, d],
            &wgpu_device,
        );
        let v_w = Tensor::from_slice(
            &vec![0.0f32; b * kv_heads * max_seq * d],
            &[b, kv_heads, max_seq, d],
            &wgpu_device,
        );
        let nk = Tensor::from_slice(
            &new_k.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &wgpu_device,
        );
        let nv = Tensor::from_slice(
            &new_v.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &wgpu_device,
        );
        wgpu_client
            .kv_cache_update(&k_w, &v_w, &nk, &nv, position)
            .unwrap();
        assert_parity_f32(
            &k_w.to_vec::<f32>(),
            &cpu_k,
            "kv_cache_update K WGPU vs CPU",
        );
        assert_parity_f32(
            &v_w.to_vec::<f32>(),
            &cpu_v,
            "kv_cache_update V WGPU vs CPU",
        );
    });
}

#[test]
fn test_reshape_and_cache_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 4;
    let num_heads = 2;
    let d = 16;
    let block_size = 4;
    let num_blocks = 2;

    let key = det_tensor(&[num_tokens, num_heads, d], &cpu_device);
    let value = det_tensor(&[num_tokens, num_heads, d], &cpu_device);
    let zeros = vec![0.0f32; num_blocks * block_size * num_heads * d];
    let key_cache = numr::tensor::Tensor::from_slice(
        &zeros,
        &[num_blocks, block_size, num_heads, d],
        &cpu_device,
    );
    let value_cache = numr::tensor::Tensor::from_slice(
        &zeros,
        &[num_blocks, block_size, num_heads, d],
        &cpu_device,
    );
    // Slot mapping: tokens go into slots 0,1,4,5 (block 0 slots 0-1, block 1 slots 0-1)
    let slot_data: Vec<i64> = vec![0, 1, 4, 5];
    let slot_mapping = numr::tensor::Tensor::from_slice(&slot_data, &[num_tokens], &cpu_device);

    cpu_client
        .reshape_and_cache(
            &key,
            &value,
            &key_cache,
            &value_cache,
            &slot_mapping,
            block_size,
        )
        .unwrap();
    let cpu_kc = key_cache.to_vec::<f32>();
    let cpu_vc = value_cache.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k = Tensor::from_slice(
            &key.to_vec::<f32>(),
            &[num_tokens, num_heads, d],
            &cuda_device,
        );
        let v = Tensor::from_slice(
            &value.to_vec::<f32>(),
            &[num_tokens, num_heads, d],
            &cuda_device,
        );
        let kc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &cuda_device,
        );
        let vc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &cuda_device,
        );
        let sm = Tensor::from_slice(&slot_data, &[num_tokens], &cuda_device);
        cuda_client
            .reshape_and_cache(&k, &v, &kc, &vc, &sm, block_size)
            .unwrap();
        assert_parity_f32(
            &kc.to_vec::<f32>(),
            &cpu_kc,
            "reshape_and_cache K CUDA vs CPU",
        );
        assert_parity_f32(
            &vc.to_vec::<f32>(),
            &cpu_vc,
            "reshape_and_cache V CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k = Tensor::from_slice(
            &key.to_vec::<f32>(),
            &[num_tokens, num_heads, d],
            &wgpu_device,
        );
        let v = Tensor::from_slice(
            &value.to_vec::<f32>(),
            &[num_tokens, num_heads, d],
            &wgpu_device,
        );
        let kc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &wgpu_device,
        );
        let vc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &wgpu_device,
        );
        let sm = Tensor::from_slice(&slot_data, &[num_tokens], &wgpu_device);
        wgpu_client
            .reshape_and_cache(&k, &v, &kc, &vc, &sm, block_size)
            .unwrap();
        assert_parity_f32(
            &kc.to_vec::<f32>(),
            &cpu_kc,
            "reshape_and_cache K WGPU vs CPU",
        );
        assert_parity_f32(
            &vc.to_vec::<f32>(),
            &cpu_vc,
            "reshape_and_cache V WGPU vs CPU",
        );
    });
}
