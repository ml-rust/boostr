//! Backend parity tests for KvCacheQuantOps.

use super::helpers::*;
use boostr::ops::traits::cache::kv_cache_quant::{Int4GroupSize, KvCacheQuantOps};

#[test]
fn test_quantize_dequantize_fp8_roundtrip_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 32;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    let (quantized, scales) = cpu_client
        .quantize_kv_fp8_per_token(&input, num_tokens, head_dim)
        .unwrap();
    let cpu_deq = cpu_client
        .dequantize_kv_fp8_per_token(
            &quantized,
            &scales,
            num_tokens,
            head_dim,
            numr::dtype::DType::F32,
        )
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::tensor::Tensor;
        let inp = Tensor::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        );
        let (q, s) = cuda_client
            .quantize_kv_fp8_per_token(&inp, num_tokens, head_dim)
            .unwrap();
        let deq = cuda_client
            .dequantize_kv_fp8_per_token(&q, &s, num_tokens, head_dim, numr::dtype::DType::F32)
            .unwrap();
        // FP8 quantization is inherently lossy; CUDA path goes F32→F16→FP8→F16→F32
        // while CPU does F32→FP8→F32 directly, so wider tolerance needed
        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "fp8 roundtrip CUDA vs CPU",
            0.1,  // 10% relative — FP8 has only 3 mantissa bits
            0.01, // absolute tolerance for values near zero
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::tensor::Tensor;
        let inp = Tensor::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &wgpu_device,
        );
        let (q, s) = wgpu_client
            .quantize_kv_fp8_per_token(&inp, num_tokens, head_dim)
            .unwrap();
        let deq = wgpu_client
            .dequantize_kv_fp8_per_token(&q, &s, num_tokens, head_dim, numr::dtype::DType::F32)
            .unwrap();
        assert_parity_f32(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "fp8 roundtrip WGPU vs CPU",
        );
    });
}

#[test]
fn test_quantize_dequantize_int4_roundtrip_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 64;
    let group_size = Int4GroupSize::Group64;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    let (packed, scales, zeros) = cpu_client
        .quantize_kv_int4(&input, num_tokens, head_dim, group_size)
        .unwrap();
    let cpu_deq = cpu_client
        .dequantize_kv_int4(&packed, &scales, &zeros, num_tokens, head_dim, group_size)
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::tensor::Tensor;
        let inp = Tensor::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        );
        let (p, s, z) = cuda_client
            .quantize_kv_int4(&inp, num_tokens, head_dim, group_size)
            .unwrap();
        let deq = cuda_client
            .dequantize_kv_int4(&p, &s, &z, num_tokens, head_dim, group_size)
            .unwrap();
        assert_parity_f32(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "int4 roundtrip CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::tensor::Tensor;
        let inp = Tensor::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &wgpu_device,
        );
        let (p, s, z) = wgpu_client
            .quantize_kv_int4(&inp, num_tokens, head_dim, group_size)
            .unwrap();
        let deq = wgpu_client
            .dequantize_kv_int4(&p, &s, &z, num_tokens, head_dim, group_size)
            .unwrap();
        assert_parity_f32(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "int4 roundtrip WGPU vs CPU",
        );
    });
}

#[test]
fn test_quantize_dequantize_int8_roundtrip_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 32;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    let (quantized, scales) = cpu_client
        .quantize_kv_int8(&input, num_tokens, head_dim)
        .unwrap();
    let cpu_deq = cpu_client
        .dequantize_kv_int8(&quantized, &scales, num_tokens, head_dim)
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::tensor::Tensor;
        let inp = Tensor::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        );
        let (q, s) = cuda_client
            .quantize_kv_int8(&inp, num_tokens, head_dim)
            .unwrap();
        let deq = cuda_client
            .dequantize_kv_int8(&q, &s, num_tokens, head_dim)
            .unwrap();
        assert_parity_f32(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "int8 roundtrip CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::tensor::Tensor;
        let inp = Tensor::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &wgpu_device,
        );
        let (q, s) = wgpu_client
            .quantize_kv_int8(&inp, num_tokens, head_dim)
            .unwrap();
        let deq = wgpu_client
            .dequantize_kv_int8(&q, &s, num_tokens, head_dim)
            .unwrap();
        assert_parity_f32(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "int8 roundtrip WGPU vs CPU",
        );
    });
}
