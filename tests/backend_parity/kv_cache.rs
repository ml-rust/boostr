//! Backend parity tests for KvCacheOps.

use super::helpers::*;
use boostr::ops::traits::cache::kv_cache::KvCacheOps;

// ============================================================================
// copy_blocks / swap_blocks fixtures
//
// `BLOCK_HEAD_DIM = 66` is a multiple of neither kernel's vec_size (4 for
// F32, 8 for F16/BF16): 66 = 16*4 + 2 = 8*8 + 2. Every test below therefore
// exercises both the vectorized copy and the scalar `tid == 0` remainder
// loop, which is the most likely place for an indexing bug in a kernel that
// has never run before.
// ============================================================================

const BLOCK_NUM_HEADS: usize = 2;
const BLOCK_HEAD_DIM: usize = 66;
const BLOCK_SIZE: usize = 3;

fn block_cache_shape(num_blocks: usize) -> [usize; 4] {
    [num_blocks, BLOCK_SIZE, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM]
}

/// Deterministic F32 fixture. `seed` shifts the phase so two calls (e.g. for
/// a key vs. a value cache) produce different but reproducible data.
fn det_f32(shape: &[usize], seed: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.137).sin() * 0.6)
        .collect()
}

#[test]
fn test_kv_cache_update_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 16, 32);
    let new_len = 4;
    let position = 3;

    // Zero-init caches
    let zeros = vec![0.0f32; b * kv_heads * max_seq * d];
    let k_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    let v_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
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
        )
        .unwrap();
        let v_c = Tensor::from_slice(
            &vec![0.0f32; b * kv_heads * max_seq * d],
            &[b, kv_heads, max_seq, d],
            &cuda_device,
        )
        .unwrap();
        let nk = Tensor::from_slice(
            &new_k.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
        let nv = Tensor::from_slice(
            &new_v.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
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
        )
        .unwrap();
        let v_w = Tensor::from_slice(
            &vec![0.0f32; b * kv_heads * max_seq * d],
            &[b, kv_heads, max_seq, d],
            &wgpu_device,
        )
        .unwrap();
        let nk = Tensor::from_slice(
            &new_k.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &wgpu_device,
        )
        .unwrap();
        let nv = Tensor::from_slice(
            &new_v.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &wgpu_device,
        )
        .unwrap();
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

/// A `new_k`/`new_v` whose head_dim disagrees with the cache must be
/// rejected identically by both backends — this is the case that drives the
/// offset math in the CUDA kernel, so a missed check here is an OOB write.
#[test]
fn test_kv_cache_update_head_dim_mismatch_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 8, 4);
    let position = 0;

    let zeros = vec![0.0f32; b * kv_heads * max_seq * d];
    let k_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    let v_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    // new_k/new_v head_dim (d+1) disagrees with the cache's head_dim (d).
    let new_k = det_tensor(&[b, kv_heads, 2, d + 1], &cpu_device);
    let new_v = det_tensor(&[b, kv_heads, 2, d + 1], &cpu_device);

    let result = cpu_client.kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, position);
    assert!(result.is_err(), "CPU: head_dim mismatch must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k_c = Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v_c = Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk = Tensor::from_slice(
            &new_k.to_vec::<f32>(),
            &[b, kv_heads, 2, d + 1],
            &cuda_device,
        )
        .unwrap();
        let nv = Tensor::from_slice(
            &new_v.to_vec::<f32>(),
            &[b, kv_heads, 2, d + 1],
            &cuda_device,
        )
        .unwrap();
        let result = cuda_client.kv_cache_update(&k_c, &v_c, &nk, &nv, position);
        assert!(result.is_err(), "CUDA: head_dim mismatch must be rejected");
    });
}

/// A `new_k`/`new_v` batch dimension disagreeing with the cache must be
/// rejected identically by both backends.
#[test]
fn test_kv_cache_update_batch_mismatch_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (2, 2, 8, 4);
    let position = 0;

    let zeros = vec![0.0f32; b * kv_heads * max_seq * d];
    let k_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    let v_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    // new_k/new_v batch (b-1) disagrees with the cache's batch (b).
    let new_k = det_tensor(&[b - 1, kv_heads, 2, d], &cpu_device);
    let new_v = det_tensor(&[b - 1, kv_heads, 2, d], &cpu_device);

    let result = cpu_client.kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, position);
    assert!(result.is_err(), "CPU: batch mismatch must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k_c = Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v_c = Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk = Tensor::from_slice(
            &new_k.to_vec::<f32>(),
            &[b - 1, kv_heads, 2, d],
            &cuda_device,
        )
        .unwrap();
        let nv = Tensor::from_slice(
            &new_v.to_vec::<f32>(),
            &[b - 1, kv_heads, 2, d],
            &cuda_device,
        )
        .unwrap();
        let result = cuda_client.kv_cache_update(&k_c, &v_c, &nk, &nv, position);
        assert!(result.is_err(), "CUDA: batch mismatch must be rejected");
    });
}

/// A dtype mismatch among the four tensors must be rejected identically by
/// both backends — CPU derives its byte-copy width from `k_cache`'s dtype
/// alone, so a silent mismatch would miscopy rather than error.
#[test]
fn test_kv_cache_update_dtype_mismatch_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 8, 4);
    let position = 0;

    let zeros = vec![0.0f32; b * kv_heads * max_seq * d];
    let k_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    let v_cache =
        numr::tensor::Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cpu_device).unwrap();
    let new_k = det_tensor(&[b, kv_heads, 2, d], &cpu_device);
    // new_v is F16 while k_cache, v_cache, and new_k are F32.
    let new_v = det_tensor(&[b, kv_heads, 2, d], &cpu_device)
        .to_dtype(numr::dtype::DType::F16)
        .unwrap();

    let result = cpu_client.kv_cache_update(&k_cache, &v_cache, &new_k, &new_v, position);
    assert!(result.is_err(), "CPU: dtype mismatch must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k_c = Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v_c = Tensor::from_slice(&zeros, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk =
            Tensor::from_slice(&new_k.to_vec::<f32>(), &[b, kv_heads, 2, d], &cuda_device).unwrap();
        let nv_f32 = new_v
            .to_dtype(numr::dtype::DType::F32)
            .unwrap()
            .to_vec::<f32>();
        let nv = Tensor::from_slice(&nv_f32, &[b, kv_heads, 2, d], &cuda_device)
            .unwrap()
            .to_dtype(numr::dtype::DType::F16)
            .unwrap();
        let result = cuda_client.kv_cache_update(&k_c, &v_c, &nk, &nv, position);
        assert!(result.is_err(), "CUDA: dtype mismatch must be rejected");
    });
}

/// CPU-vs-CUDA parity for `kv_cache_update_batched`, checked against the
/// already-trusted `kv_cache_update` called once per layer.
///
/// Uses >=3 layers with different data per layer (a kernel that ignores the
/// `y` grid dimension, or overwrites layer 0 repeatedly, would still pass a
/// single-layer test), a non-zero `position`, and `new_len > 1` so the
/// offset arithmetic is exercised.
#[cfg(feature = "cuda")]
fn assert_kv_cache_update_batched_parity(dtype: numr::dtype::DType, label: &str) {
    use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
    use numr::tensor::Tensor;

    let (b, kv_heads, max_seq, d) = (2, 3, 16, 8);
    let new_len = 3;
    let position = 5;
    let num_layers = 4;

    let (cpu_client, cpu_device) = setup_cpu();

    let cache_shape = [b, kv_heads, max_seq, d];
    let new_shape = [b, kv_heads, new_len, d];
    let zeros_cache = vec![0.0f32; b * kv_heads * max_seq * d];

    // Distinct data per layer via a per-layer seed, so a layer-index bug
    // (wrong y-index, or every layer reading layer 0's pointers) is caught.
    let new_k_data: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| det_f32_seeded(&new_shape, l as f32 * 17.0))
        .collect();
    let new_v_data: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| det_f32_seeded(&new_shape, l as f32 * 17.0 + 500.0))
        .collect();

    // Oracle: call the trusted single-layer kv_cache_update once per layer.
    let mut oracle_k = Vec::with_capacity(num_layers);
    let mut oracle_v = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let kc = Tensor::from_slice(&zeros_cache, &cache_shape, &cpu_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let vc = Tensor::from_slice(&zeros_cache, &cache_shape, &cpu_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let nk = Tensor::from_slice(&new_k_data[l], &new_shape, &cpu_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let nv = Tensor::from_slice(&new_v_data[l], &new_shape, &cpu_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        cpu_client
            .kv_cache_update(&kc, &vc, &nk, &nv, position)
            .unwrap();
        oracle_k.push(
            kc.to_dtype(numr::dtype::DType::F32)
                .unwrap()
                .to_vec::<f32>(),
        );
        oracle_v.push(
            vc.to_dtype(numr::dtype::DType::F32)
                .unwrap()
                .to_vec::<f32>(),
        );
    }

    // CPU batched result, compared against the oracle.
    let cpu_k_caches: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = (0..num_layers)
        .map(|_| {
            Tensor::from_slice(&zeros_cache, &cache_shape, &cpu_device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        })
        .collect();
    let cpu_v_caches: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = (0..num_layers)
        .map(|_| {
            Tensor::from_slice(&zeros_cache, &cache_shape, &cpu_device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        })
        .collect();
    let cpu_new_ks: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = (0..num_layers)
        .map(|l| {
            Tensor::from_slice(&new_k_data[l], &new_shape, &cpu_device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        })
        .collect();
    let cpu_new_vs: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = (0..num_layers)
        .map(|l| {
            Tensor::from_slice(&new_v_data[l], &new_shape, &cpu_device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        })
        .collect();

    let kc_refs: Vec<&Tensor<_>> = cpu_k_caches.iter().collect();
    let vc_refs: Vec<&Tensor<_>> = cpu_v_caches.iter().collect();
    let nk_refs: Vec<&Tensor<_>> = cpu_new_ks.iter().collect();
    let nv_refs: Vec<&Tensor<_>> = cpu_new_vs.iter().collect();

    cpu_client
        .kv_cache_update_batched(&kc_refs, &vc_refs, &nk_refs, &nv_refs, max_seq, position)
        .unwrap();

    for l in 0..num_layers {
        assert_eq!(
            cpu_k_caches[l]
                .to_dtype(numr::dtype::DType::F32)
                .unwrap()
                .to_vec::<f32>(),
            oracle_k[l],
            "{label} CPU batched layer {l} K vs per-layer oracle"
        );
        assert_eq!(
            cpu_v_caches[l]
                .to_dtype(numr::dtype::DType::F32)
                .unwrap()
                .to_vec::<f32>(),
            oracle_v[l],
            "{label} CPU batched layer {l} V vs per-layer oracle"
        );
    }

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_k_caches: Vec<Tensor<numr::runtime::cuda::CudaRuntime>> = (0..num_layers)
            .map(|_| {
                Tensor::from_slice(&zeros_cache, &cache_shape, &cuda_device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
            })
            .collect();
        let cuda_v_caches: Vec<Tensor<numr::runtime::cuda::CudaRuntime>> = (0..num_layers)
            .map(|_| {
                Tensor::from_slice(&zeros_cache, &cache_shape, &cuda_device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
            })
            .collect();
        let cuda_new_ks: Vec<Tensor<numr::runtime::cuda::CudaRuntime>> = (0..num_layers)
            .map(|l| {
                Tensor::from_slice(&new_k_data[l], &new_shape, &cuda_device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
            })
            .collect();
        let cuda_new_vs: Vec<Tensor<numr::runtime::cuda::CudaRuntime>> = (0..num_layers)
            .map(|l| {
                Tensor::from_slice(&new_v_data[l], &new_shape, &cuda_device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
            })
            .collect();

        let kc_refs: Vec<&Tensor<_>> = cuda_k_caches.iter().collect();
        let vc_refs: Vec<&Tensor<_>> = cuda_v_caches.iter().collect();
        let nk_refs: Vec<&Tensor<_>> = cuda_new_ks.iter().collect();
        let nv_refs: Vec<&Tensor<_>> = cuda_new_vs.iter().collect();

        cuda_client
            .kv_cache_update_batched(&kc_refs, &vc_refs, &nk_refs, &nv_refs, max_seq, position)
            .unwrap();

        for l in 0..num_layers {
            assert_eq!(
                cuda_k_caches[l]
                    .to_dtype(numr::dtype::DType::F32)
                    .unwrap()
                    .to_vec::<f32>(),
                oracle_k[l],
                "{label} CUDA batched layer {l} K vs per-layer oracle"
            );
            assert_eq!(
                cuda_v_caches[l]
                    .to_dtype(numr::dtype::DType::F32)
                    .unwrap()
                    .to_vec::<f32>(),
                oracle_v[l],
                "{label} CUDA batched layer {l} V vs per-layer oracle"
            );
        }
    });
}

/// Deterministic F32 fixture matching `det_f32`'s formula, taking an
/// explicit shape slice (used for per-layer seeding above).
fn det_f32_seeded(shape: &[usize], seed: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.137).sin() * 0.6)
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn test_kv_cache_update_batched_f32_parity() {
    assert_kv_cache_update_batched_parity(numr::dtype::DType::F32, "kv_cache_update_batched f32");
}

#[cfg(feature = "cuda")]
#[test]
fn test_kv_cache_update_batched_f16_parity() {
    assert_kv_cache_update_batched_parity(numr::dtype::DType::F16, "kv_cache_update_batched f16");
}

#[cfg(feature = "cuda")]
#[test]
fn test_kv_cache_update_batched_bf16_parity() {
    assert_kv_cache_update_batched_parity(numr::dtype::DType::BF16, "kv_cache_update_batched bf16");
}

#[test]
fn test_kv_cache_update_batched_mismatched_lengths_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 8, 4);
    let new_len = 2;
    let position = 0;

    let zeros_cache = vec![0.0f32; b * kv_heads * max_seq * d];
    let k0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let k1 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let v0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let nk0 = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);
    let nv0 = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);

    // 2 K caches but only 1 V cache: lengths must match.
    let result = cpu_client.kv_cache_update_batched(
        &[&k0, &k1],
        &[&v0],
        &[&nk0, &nk0],
        &[&nv0, &nv0],
        max_seq,
        position,
    );
    assert!(result.is_err(), "mismatched slice lengths must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let k1 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk0 = Tensor::from_slice(
            &nk0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
        let nv0 = Tensor::from_slice(
            &nv0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
        let result = cuda_client.kv_cache_update_batched(
            &[&k0, &k1],
            &[&v0],
            &[&nk0, &nk0],
            &[&nv0, &nv0],
            max_seq,
            position,
        );
        assert!(
            result.is_err(),
            "CUDA: mismatched slice lengths must be rejected"
        );
    });
}

/// The batched CUDA path validates cache-vs-new_k/new_v compatibility
/// directly (it does not delegate to the single-layer `kv_cache_update`),
/// so it needs its own head_dim-mismatch coverage.
#[test]
fn test_kv_cache_update_batched_head_dim_mismatch_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 8, 4);
    let new_len = 2;
    let position = 0;

    let zeros_cache = vec![0.0f32; b * kv_heads * max_seq * d];
    let k0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let v0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    // new_k/new_v head_dim (d+1) disagrees with the cache's head_dim (d).
    let nk0 = det_tensor(&[b, kv_heads, new_len, d + 1], &cpu_device);
    let nv0 = det_tensor(&[b, kv_heads, new_len, d + 1], &cpu_device);

    let result =
        cpu_client.kv_cache_update_batched(&[&k0], &[&v0], &[&nk0], &[&nv0], max_seq, position);
    assert!(result.is_err(), "CPU: head_dim mismatch must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk0 = Tensor::from_slice(
            &nk0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d + 1],
            &cuda_device,
        )
        .unwrap();
        let nv0 = Tensor::from_slice(
            &nv0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d + 1],
            &cuda_device,
        )
        .unwrap();
        let result = cuda_client.kv_cache_update_batched(
            &[&k0],
            &[&v0],
            &[&nk0],
            &[&nv0],
            max_seq,
            position,
        );
        assert!(result.is_err(), "CUDA: head_dim mismatch must be rejected");
    });
}

/// A dtype mismatch among the four tensors must be rejected identically by
/// both backends through the batched path too.
#[test]
fn test_kv_cache_update_batched_dtype_mismatch_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 8, 4);
    let new_len = 2;
    let position = 0;

    let zeros_cache = vec![0.0f32; b * kv_heads * max_seq * d];
    let k0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let v0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let nk0 = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);
    // nv0 is F16 while k0, v0, and nk0 are F32.
    let nv0 = det_tensor(&[b, kv_heads, new_len, d], &cpu_device)
        .to_dtype(numr::dtype::DType::F16)
        .unwrap();

    let result =
        cpu_client.kv_cache_update_batched(&[&k0], &[&v0], &[&nk0], &[&nv0], max_seq, position);
    assert!(result.is_err(), "CPU: dtype mismatch must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk0 = Tensor::from_slice(
            &nk0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
        let nv0_f32 = nv0
            .to_dtype(numr::dtype::DType::F32)
            .unwrap()
            .to_vec::<f32>();
        let nv0 = Tensor::from_slice(&nv0_f32, &[b, kv_heads, new_len, d], &cuda_device)
            .unwrap()
            .to_dtype(numr::dtype::DType::F16)
            .unwrap();
        let result = cuda_client.kv_cache_update_batched(
            &[&k0],
            &[&v0],
            &[&nk0],
            &[&nv0],
            max_seq,
            position,
        );
        assert!(result.is_err(), "CUDA: dtype mismatch must be rejected");
    });
}

#[test]
fn test_kv_cache_update_batched_position_overflow_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, kv_heads, max_seq, d) = (1, 2, 4, 4);
    let new_len = 2;
    let position = 3; // 3 + 2 > 4

    let zeros_cache = vec![0.0f32; b * kv_heads * max_seq * d];
    let k0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let v0 =
        numr::tensor::Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cpu_device)
            .unwrap();
    let nk0 = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);
    let nv0 = det_tensor(&[b, kv_heads, new_len, d], &cpu_device);

    let result =
        cpu_client.kv_cache_update_batched(&[&k0], &[&v0], &[&nk0], &[&nv0], max_seq, position);
    assert!(
        result.is_err(),
        "position + new_len > max_seq_len must be rejected"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let k0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let v0 =
            Tensor::from_slice(&zeros_cache, &[b, kv_heads, max_seq, d], &cuda_device).unwrap();
        let nk0 = Tensor::from_slice(
            &nk0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
        let nv0 = Tensor::from_slice(
            &nv0.to_vec::<f32>(),
            &[b, kv_heads, new_len, d],
            &cuda_device,
        )
        .unwrap();
        let result = cuda_client.kv_cache_update_batched(
            &[&k0],
            &[&v0],
            &[&nk0],
            &[&nv0],
            max_seq,
            position,
        );
        assert!(
            result.is_err(),
            "CUDA: position + new_len > max_seq_len must be rejected"
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
    )
    .unwrap();
    let value_cache = numr::tensor::Tensor::from_slice(
        &zeros,
        &[num_blocks, block_size, num_heads, d],
        &cpu_device,
    )
    .unwrap();
    // Slot mapping: tokens go into slots 0,1,4,5 (block 0 slots 0-1, block 1 slots 0-1)
    let slot_data: Vec<i32> = vec![0, 1, 4, 5];
    let slot_mapping =
        numr::tensor::Tensor::from_slice(&slot_data, &[num_tokens], &cpu_device).unwrap();

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
        )
        .unwrap();
        let v = Tensor::from_slice(
            &value.to_vec::<f32>(),
            &[num_tokens, num_heads, d],
            &cuda_device,
        )
        .unwrap();
        let kc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &cuda_device,
        )
        .unwrap();
        let vc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &cuda_device,
        )
        .unwrap();
        let sm = Tensor::from_slice(&slot_data, &[num_tokens], &cuda_device).unwrap();
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
        )
        .unwrap();
        let v = Tensor::from_slice(
            &value.to_vec::<f32>(),
            &[num_tokens, num_heads, d],
            &wgpu_device,
        )
        .unwrap();
        let kc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &wgpu_device,
        )
        .unwrap();
        let vc = Tensor::from_slice(
            &vec![0.0f32; num_blocks * block_size * num_heads * d],
            &[num_blocks, block_size, num_heads, d],
            &wgpu_device,
        )
        .unwrap();
        let sm = Tensor::from_slice(&slot_data, &[num_tokens], &wgpu_device).unwrap();
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

#[test]
fn test_copy_blocks_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    // 3 pairs: src blocks 0,1,2 each copied into a distinct dst block.
    let mapping_data: Vec<i32> = vec![0, 3, 1, 4, 2, 5];
    let shape = block_cache_shape(6);

    let key_data = det_f32(&shape, 0.0);
    let value_data = det_f32(&shape, 100.0);
    let key_cache = numr::tensor::Tensor::from_slice(&key_data, &shape, &cpu_device).unwrap();
    let value_cache = numr::tensor::Tensor::from_slice(&value_data, &shape, &cpu_device).unwrap();
    let block_mapping = det_i32_tensor(&mapping_data, &[mapping_data.len()], &cpu_device);

    cpu_client
        .copy_blocks(
            &key_cache,
            &value_cache,
            &block_mapping,
            BLOCK_NUM_HEADS,
            BLOCK_HEAD_DIM,
            BLOCK_SIZE,
        )
        .unwrap();

    let key_after = key_cache.to_vec::<f32>();
    let value_after = value_cache.to_vec::<f32>();
    let block_stride = BLOCK_SIZE * BLOCK_NUM_HEADS * BLOCK_HEAD_DIM;

    for pair in mapping_data.chunks(2) {
        let (src, dst) = (pair[0] as usize, pair[1] as usize);
        assert_eq!(
            key_after[dst * block_stride..(dst + 1) * block_stride],
            key_data[src * block_stride..(src + 1) * block_stride],
            "copy_blocks CPU: key block {src} -> {dst} mismatch"
        );
        assert_eq!(
            value_after[dst * block_stride..(dst + 1) * block_stride],
            value_data[src * block_stride..(src + 1) * block_stride],
            "copy_blocks CPU: value block {src} -> {dst} mismatch"
        );
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let kc = Tensor::from_slice(&key_data, &shape, &cuda_device).unwrap();
        let vc = Tensor::from_slice(&value_data, &shape, &cuda_device).unwrap();
        let bm = Tensor::from_slice(&mapping_data, &[mapping_data.len()], &cuda_device).unwrap();
        cuda_client
            .copy_blocks(&kc, &vc, &bm, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM, BLOCK_SIZE)
            .unwrap();
        // Pure data movement, no arithmetic: CUDA must match CPU bit-for-bit,
        // so this compares exactly rather than with a tolerance.
        assert_eq!(kc.to_vec::<f32>(), key_after, "copy_blocks key CUDA vs CPU");
        assert_eq!(
            vc.to_vec::<f32>(),
            value_after,
            "copy_blocks value CUDA vs CPU"
        );
    });
}

#[test]
fn test_swap_blocks_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    // 3 pairs, including a self-pair (2 -> 2): valid since src_cache and
    // dst_cache are two distinct buffers, never aliased.
    let mapping_data: Vec<i32> = vec![0, 3, 1, 0, 2, 2];
    let shape = block_cache_shape(4);

    let src_data = det_f32(&shape, 7.0);
    let dst_data_init = det_f32(&shape, 900.0);
    let src_cache = numr::tensor::Tensor::from_slice(&src_data, &shape, &cpu_device).unwrap();
    let dst_cache = numr::tensor::Tensor::from_slice(&dst_data_init, &shape, &cpu_device).unwrap();
    let block_mapping = det_i32_tensor(&mapping_data, &[mapping_data.len()], &cpu_device);

    cpu_client
        .swap_blocks(
            &src_cache,
            &dst_cache,
            &block_mapping,
            BLOCK_NUM_HEADS,
            BLOCK_HEAD_DIM,
            BLOCK_SIZE,
        )
        .unwrap();

    let dst_after = dst_cache.to_vec::<f32>();
    let src_after = src_cache.to_vec::<f32>();
    let block_stride = BLOCK_SIZE * BLOCK_NUM_HEADS * BLOCK_HEAD_DIM;

    assert_eq!(
        src_after, src_data,
        "swap_blocks CPU: src_cache must not be mutated"
    );
    for pair in mapping_data.chunks(2) {
        let (src, dst) = (pair[0] as usize, pair[1] as usize);
        assert_eq!(
            dst_after[dst * block_stride..(dst + 1) * block_stride],
            src_data[src * block_stride..(src + 1) * block_stride],
            "swap_blocks CPU: block {src} -> {dst} mismatch"
        );
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let sc = Tensor::from_slice(&src_data, &shape, &cuda_device).unwrap();
        let dc = Tensor::from_slice(&dst_data_init, &shape, &cuda_device).unwrap();
        let bm = Tensor::from_slice(&mapping_data, &[mapping_data.len()], &cuda_device).unwrap();
        cuda_client
            .swap_blocks(&sc, &dc, &bm, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM, BLOCK_SIZE)
            .unwrap();
        assert_eq!(dc.to_vec::<f32>(), dst_after, "swap_blocks dst CUDA vs CPU");
    });
}

#[test]
fn test_copy_blocks_odd_mapping_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let shape = block_cache_shape(4);
    let key_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cpu_device).unwrap();
    let value_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cpu_device).unwrap();
    // 3 entries cannot split into src/dst pairs.
    let block_mapping = det_i32_tensor(&[0, 1, 2], &[3], &cpu_device);

    let result = cpu_client.copy_blocks(
        &key_cache,
        &value_cache,
        &block_mapping,
        BLOCK_NUM_HEADS,
        BLOCK_HEAD_DIM,
        BLOCK_SIZE,
    );
    assert!(result.is_err(), "odd block_mapping length must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let kc = Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cuda_device).unwrap();
        let vc = Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cuda_device).unwrap();
        let bm = Tensor::from_slice(&[0i32, 1, 2], &[3], &cuda_device).unwrap();
        let result =
            cuda_client.copy_blocks(&kc, &vc, &bm, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM, BLOCK_SIZE);
        assert!(
            result.is_err(),
            "CUDA: odd block_mapping length must be rejected"
        );
    });
}

// An empty block_mapping means there is no work to do, so both backends take
// an early return. An unsupported cache dtype must still be rejected on that
// path: CUDA once returned Ok here while CPU returned Err, so the same call
// answered differently per backend.
#[test]
fn test_block_ops_reject_bad_dtype_with_empty_mapping() {
    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [4, BLOCK_SIZE, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM];
    let n: usize = shape.iter().product();

    let kc = det_i32_tensor(&vec![0i32; n], &shape, &cpu_device);
    let vc = det_i32_tensor(&vec![0i32; n], &shape, &cpu_device);
    let empty = det_i32_tensor(&[], &[0], &cpu_device);

    let result = cpu_client.copy_blocks(
        &kc,
        &vc,
        &empty,
        BLOCK_NUM_HEADS,
        BLOCK_HEAD_DIM,
        BLOCK_SIZE,
    );
    assert!(
        result.is_err(),
        "CPU: unsupported cache dtype must be rejected even with an empty mapping"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let kc = Tensor::from_slice(&vec![0i32; n], &shape, &cuda_device).unwrap();
        let vc = Tensor::from_slice(&vec![0i32; n], &shape, &cuda_device).unwrap();
        let empty = Tensor::from_slice(&[] as &[i32], &[0], &cuda_device).unwrap();
        let result = cuda_client.copy_blocks(
            &kc,
            &vc,
            &empty,
            BLOCK_NUM_HEADS,
            BLOCK_HEAD_DIM,
            BLOCK_SIZE,
        );
        assert!(
            result.is_err(),
            "CUDA: unsupported cache dtype must be rejected even with an empty mapping"
        );
    });
}

#[test]
fn test_copy_blocks_out_of_range_block_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_blocks = 4;
    let shape = block_cache_shape(num_blocks);
    let key_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cpu_device).unwrap();
    let value_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cpu_device).unwrap();
    // dst block index == num_blocks is one past the last valid block.
    let block_mapping = det_i32_tensor(&[0, num_blocks as i32], &[2], &cpu_device);

    let result = cpu_client.copy_blocks(
        &key_cache,
        &value_cache,
        &block_mapping,
        BLOCK_NUM_HEADS,
        BLOCK_HEAD_DIM,
        BLOCK_SIZE,
    );
    assert!(result.is_err(), "out-of-range block index must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let kc = Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cuda_device).unwrap();
        let vc = Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cuda_device).unwrap();
        let bm = Tensor::from_slice(&[0i32, num_blocks as i32], &[2], &cuda_device).unwrap();
        let result =
            cuda_client.copy_blocks(&kc, &vc, &bm, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM, BLOCK_SIZE);
        assert!(
            result.is_err(),
            "CUDA: out-of-range block index must be rejected"
        );
    });
}

#[test]
fn test_swap_blocks_odd_mapping_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let shape = block_cache_shape(4);
    let src_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cpu_device).unwrap();
    let dst_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cpu_device).unwrap();
    let block_mapping = det_i32_tensor(&[0, 1, 2], &[3], &cpu_device);

    let result = cpu_client.swap_blocks(
        &src_cache,
        &dst_cache,
        &block_mapping,
        BLOCK_NUM_HEADS,
        BLOCK_HEAD_DIM,
        BLOCK_SIZE,
    );
    assert!(result.is_err(), "odd block_mapping length must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let sc = Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cuda_device).unwrap();
        let dc = Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cuda_device).unwrap();
        let bm = Tensor::from_slice(&[0i32, 1, 2], &[3], &cuda_device).unwrap();
        let result =
            cuda_client.swap_blocks(&sc, &dc, &bm, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM, BLOCK_SIZE);
        assert!(
            result.is_err(),
            "CUDA: odd block_mapping length must be rejected"
        );
    });
}

#[test]
fn test_swap_blocks_out_of_range_block_is_error() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_blocks = 4;
    let shape = block_cache_shape(num_blocks);
    let src_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cpu_device).unwrap();
    let dst_cache =
        numr::tensor::Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cpu_device).unwrap();
    let block_mapping = det_i32_tensor(&[0, num_blocks as i32], &[2], &cpu_device);

    let result = cpu_client.swap_blocks(
        &src_cache,
        &dst_cache,
        &block_mapping,
        BLOCK_NUM_HEADS,
        BLOCK_HEAD_DIM,
        BLOCK_SIZE,
    );
    assert!(result.is_err(), "out-of-range block index must be rejected");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use numr::tensor::Tensor;
        let sc = Tensor::from_slice(&det_f32(&shape, 0.0), &shape, &cuda_device).unwrap();
        let dc = Tensor::from_slice(&det_f32(&shape, 1.0), &shape, &cuda_device).unwrap();
        let bm = Tensor::from_slice(&[0i32, num_blocks as i32], &[2], &cuda_device).unwrap();
        let result =
            cuda_client.swap_blocks(&sc, &dc, &bm, BLOCK_NUM_HEADS, BLOCK_HEAD_DIM, BLOCK_SIZE);
        assert!(
            result.is_err(),
            "CUDA: out-of-range block index must be rejected"
        );
    });
}

/// CPU-vs-CUDA parity for `copy_blocks` in a half dtype.
///
/// Fixtures are built in F32 and cast to `dtype` with `Tensor::to_dtype` —
/// host-side `half::f16`/`half::bf16` values are not numr `Element`s (see
/// `flash_v2_fwd_sm_halfprec_parity_cuda.rs`), and `to_dtype`'s cast kernel
/// has a working fallback with or without numr's `f16` feature, so this does
/// NOT need to be gated on it — only on `cuda`, to reach a CUDA device at all.
/// copy_blocks only moves data (no arithmetic), so casting back to F32 for
/// comparison loses nothing and the check is exact, not toleranced.
#[cfg(feature = "cuda")]
fn assert_copy_blocks_half_parity(dtype: numr::dtype::DType, label: &str) {
    use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let mapping_data: Vec<i32> = vec![0, 3, 1, 4, 2, 5];
    let shape = block_cache_shape(6);
    let key_data = det_f32(&shape, 0.0);
    let value_data = det_f32(&shape, 100.0);

    let key_cpu = Tensor::from_slice(&key_data, &shape, &cpu_device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let value_cpu = Tensor::from_slice(&value_data, &shape, &cpu_device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let mapping_cpu = det_i32_tensor(&mapping_data, &[mapping_data.len()], &cpu_device);
    cpu_client
        .copy_blocks(
            &key_cpu,
            &value_cpu,
            &mapping_cpu,
            BLOCK_NUM_HEADS,
            BLOCK_HEAD_DIM,
            BLOCK_SIZE,
        )
        .unwrap();
    let key_cpu_f32 = key_cpu
        .to_dtype(numr::dtype::DType::F32)
        .unwrap()
        .to_vec::<f32>();
    let value_cpu_f32 = value_cpu
        .to_dtype(numr::dtype::DType::F32)
        .unwrap()
        .to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        let key_cuda = Tensor::from_slice(&key_data, &shape, &cuda_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let value_cuda = Tensor::from_slice(&value_data, &shape, &cuda_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let mapping_cuda =
            Tensor::from_slice(&mapping_data, &[mapping_data.len()], &cuda_device).unwrap();
        cuda_client
            .copy_blocks(
                &key_cuda,
                &value_cuda,
                &mapping_cuda,
                BLOCK_NUM_HEADS,
                BLOCK_HEAD_DIM,
                BLOCK_SIZE,
            )
            .unwrap();
        let key_cuda_f32 = key_cuda
            .to_dtype(numr::dtype::DType::F32)
            .unwrap()
            .to_vec::<f32>();
        let value_cuda_f32 = value_cuda
            .to_dtype(numr::dtype::DType::F32)
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(key_cuda_f32, key_cpu_f32, "{label} key CUDA vs CPU");
        assert_eq!(value_cuda_f32, value_cpu_f32, "{label} value CUDA vs CPU");
    });
}

/// CPU-vs-CUDA parity for `swap_blocks` in a half dtype. See
/// `assert_copy_blocks_half_parity` for why this needs no `f16` feature gate
/// and why the comparison is exact.
#[cfg(feature = "cuda")]
fn assert_swap_blocks_half_parity(dtype: numr::dtype::DType, label: &str) {
    use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let mapping_data: Vec<i32> = vec![0, 3, 1, 0, 2, 2];
    let shape = block_cache_shape(4);
    let src_data = det_f32(&shape, 7.0);
    let dst_data_init = det_f32(&shape, 900.0);

    let src_cpu = Tensor::from_slice(&src_data, &shape, &cpu_device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let dst_cpu = Tensor::from_slice(&dst_data_init, &shape, &cpu_device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let mapping_cpu = det_i32_tensor(&mapping_data, &[mapping_data.len()], &cpu_device);
    cpu_client
        .swap_blocks(
            &src_cpu,
            &dst_cpu,
            &mapping_cpu,
            BLOCK_NUM_HEADS,
            BLOCK_HEAD_DIM,
            BLOCK_SIZE,
        )
        .unwrap();
    let dst_cpu_f32 = dst_cpu
        .to_dtype(numr::dtype::DType::F32)
        .unwrap()
        .to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        let src_cuda = Tensor::from_slice(&src_data, &shape, &cuda_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let dst_cuda = Tensor::from_slice(&dst_data_init, &shape, &cuda_device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let mapping_cuda =
            Tensor::from_slice(&mapping_data, &[mapping_data.len()], &cuda_device).unwrap();
        cuda_client
            .swap_blocks(
                &src_cuda,
                &dst_cuda,
                &mapping_cuda,
                BLOCK_NUM_HEADS,
                BLOCK_HEAD_DIM,
                BLOCK_SIZE,
            )
            .unwrap();
        let dst_cuda_f32 = dst_cuda
            .to_dtype(numr::dtype::DType::F32)
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(dst_cuda_f32, dst_cpu_f32, "{label} dst CUDA vs CPU");
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_copy_blocks_f16_parity() {
    assert_copy_blocks_half_parity(numr::dtype::DType::F16, "copy_blocks f16");
}

#[cfg(feature = "cuda")]
#[test]
fn test_copy_blocks_bf16_parity() {
    assert_copy_blocks_half_parity(numr::dtype::DType::BF16, "copy_blocks bf16");
}

#[cfg(feature = "cuda")]
#[test]
fn test_swap_blocks_f16_parity() {
    assert_swap_blocks_half_parity(numr::dtype::DType::F16, "swap_blocks f16");
}

#[cfg(feature = "cuda")]
#[test]
fn test_swap_blocks_bf16_parity() {
    assert_swap_blocks_half_parity(numr::dtype::DType::BF16, "swap_blocks bf16");
}
