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
        )
        .unwrap();
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
        )
        .unwrap();
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
fn test_quantize_dequantize_fp8_per_head_roundtrip_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 32;
    let span = seq_len * head_dim;
    let input = det_tensor(&[num_heads, seq_len, head_dim], &cpu_device);

    let (quantized, scales) = cpu_client
        .quantize_kv_fp8_per_head(&input, num_heads, seq_len, head_dim)
        .unwrap();
    assert_eq!(scales.shape(), &[num_heads]);

    // One scale per head, so the existing per-token dequantizer applies
    // directly with num_tokens = num_heads and head_dim = the head's span.
    let cpu_deq = cpu_client
        .dequantize_kv_fp8_per_token(
            &quantized,
            &scales,
            num_heads,
            span,
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
            &[num_heads, seq_len, head_dim],
            &cuda_device,
        )
        .unwrap();
        let (q, s) = cuda_client
            .quantize_kv_fp8_per_head(&inp, num_heads, seq_len, head_dim)
            .unwrap();
        assert_eq!(s.shape(), &[num_heads]);
        let deq = cuda_client
            .dequantize_kv_fp8_per_token(&q, &s, num_heads, span, numr::dtype::DType::F32)
            .unwrap();
        // FP8 quantization is inherently lossy (3 mantissa bits for e4m3).
        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "fp8 per-head roundtrip CUDA vs CPU",
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
            &[num_heads, seq_len, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let (q, s) = wgpu_client
            .quantize_kv_fp8_per_head(&inp, num_heads, seq_len, head_dim)
            .unwrap();
        assert_eq!(s.shape(), &[num_heads]);
        let deq = wgpu_client
            .dequantize_kv_fp8_per_token(&q, &s, num_heads, span, numr::dtype::DType::F32)
            .unwrap();
        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "fp8 per-head roundtrip WGPU vs CPU",
            0.1,
            0.01,
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
        .dequantize_kv_int4(
            &packed,
            &scales,
            &zeros,
            num_tokens,
            head_dim,
            group_size,
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
        )
        .unwrap();
        let (p, s, z) = cuda_client
            .quantize_kv_int4(&inp, num_tokens, head_dim, group_size)
            .unwrap();
        let deq = cuda_client
            .dequantize_kv_int4(
                &p,
                &s,
                &z,
                num_tokens,
                head_dim,
                group_size,
                numr::dtype::DType::F32,
            )
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
        )
        .unwrap();
        let (p, s, z) = wgpu_client
            .quantize_kv_int4(&inp, num_tokens, head_dim, group_size)
            .unwrap();
        let deq = wgpu_client
            .dequantize_kv_int4(
                &p,
                &s,
                &z,
                num_tokens,
                head_dim,
                group_size,
                numr::dtype::DType::F32,
            )
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
        )
        .unwrap();
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
        )
        .unwrap();
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

// Covers the bf16 INT8 kv-cache quant kernel, which runs on every supported device.
#[test]
fn test_quantize_kv_int8_bf16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 32;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    // Reference computed on CPU in F32, per house rule for BF16 fixtures.
    let (cpu_quantized, cpu_scales) = cpu_client
        .quantize_kv_int8(&input, num_tokens, head_dim)
        .unwrap();
    let cpu_deq = cpu_client
        .dequantize_kv_int8(&cpu_quantized, &cpu_scales, num_tokens, head_dim)
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let inp_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        )
        .unwrap();
        let inp_bf16 = inp_f32
            .to_dtype(DType::BF16)
            .expect("cast input fixture to BF16");

        let (quantized, scales) = cuda_client
            .quantize_kv_int8(&inp_bf16, num_tokens, head_dim)
            .expect("BF16 quantize_kv_int8 must succeed");
        let deq = cuda_client
            .dequantize_kv_int8(&quantized, &scales, num_tokens, head_dim)
            .expect("dequantize_kv_int8 must succeed after a successful BF16 quantize");

        // BF16 keeps ~8 mantissa bits, so tolerance is set by the dtype, not
        // by the op.
        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "quantize_kv_int8 BF16 CUDA vs CPU",
            4e-2,
            2e-2,
        );
    });
}

// Covers the F16 INT4 kv-cache quant kernel, which runs on every supported device.
#[test]
fn test_quantize_kv_int4_f16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 64;
    let group_size = Int4GroupSize::Group64;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    // Reference computed on CPU in F32, per house rule for narrow-dtype fixtures.
    let (cpu_packed, cpu_scales, cpu_zeros) = cpu_client
        .quantize_kv_int4(&input, num_tokens, head_dim, group_size)
        .unwrap();
    let cpu_deq = cpu_client
        .dequantize_kv_int4(
            &cpu_packed,
            &cpu_scales,
            &cpu_zeros,
            num_tokens,
            head_dim,
            group_size,
            numr::dtype::DType::F32,
        )
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let inp_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        )
        .unwrap();
        let inp_f16 = inp_f32
            .to_dtype(DType::F16)
            .expect("cast input fixture to F16");

        let (packed, scales, zeros) = cuda_client
            .quantize_kv_int4(&inp_f16, num_tokens, head_dim, group_size)
            .expect("F16 quantize_kv_int4 must succeed");
        let deq = cuda_client
            .dequantize_kv_int4(
                &packed,
                &scales,
                &zeros,
                num_tokens,
                head_dim,
                group_size,
                DType::F32,
            )
            .expect("dequantize_kv_int4 must succeed after a successful F16 quantize");

        // INT4 has 16 levels. This fixture spans about 1.0 per Group64 group,
        // so the quantization step is about 0.067. F16 input rounding is far
        // smaller than that step, but near a bucket edge it can flip which
        // bucket a value lands in, so the round-trip can differ by up to one
        // step. atol is set just above one step; rtol stays small since the
        // error is bucket-sized, not proportional to the value.
        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "quantize_kv_int4 F16 CUDA vs CPU",
            2e-2,
            7e-2,
        );
    });
}

// Covers the BF16 INT4 kv-cache quant kernel, which runs on every supported device.
#[test]
fn test_quantize_kv_int4_bf16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 64;
    let group_size = Int4GroupSize::Group64;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    // Reference computed on CPU in F32, per house rule for narrow-dtype fixtures.
    let (cpu_packed, cpu_scales, cpu_zeros) = cpu_client
        .quantize_kv_int4(&input, num_tokens, head_dim, group_size)
        .unwrap();
    let cpu_deq = cpu_client
        .dequantize_kv_int4(
            &cpu_packed,
            &cpu_scales,
            &cpu_zeros,
            num_tokens,
            head_dim,
            group_size,
            numr::dtype::DType::F32,
        )
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let inp_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        )
        .unwrap();
        let inp_bf16 = inp_f32
            .to_dtype(DType::BF16)
            .expect("cast input fixture to BF16");

        let (packed, scales, zeros) = cuda_client
            .quantize_kv_int4(&inp_bf16, num_tokens, head_dim, group_size)
            .expect("BF16 quantize_kv_int4 must succeed");
        let deq = cuda_client
            .dequantize_kv_int4(
                &packed,
                &scales,
                &zeros,
                num_tokens,
                head_dim,
                group_size,
                DType::F32,
            )
            .expect("dequantize_kv_int4 must succeed after a successful BF16 quantize");

        // INT4 has 16 levels. This fixture spans about 1.0 per Group64 group,
        // so the quantization step is about 0.067. BF16 input rounding is
        // coarser than F16 but still far smaller than that step; the same
        // one-bucket boundary flip risk applies, so the tolerance matches
        // the F16 case rather than widening further.
        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "quantize_kv_int4 BF16 CUDA vs CPU",
            2e-2,
            7e-2,
        );
    });
}

// Covers dequantize_kv_fp8_per_token's F16 and BF16 output-dtype kernels,
// which run on every supported device.
#[test]
fn test_dequantize_kv_fp8_narrow_output_dtypes() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 32;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    let (quantized, scales) = cpu_client
        .quantize_kv_fp8_per_token(&input, num_tokens, head_dim)
        .unwrap();
    let cpu_deq_f32 = cpu_client
        .dequantize_kv_fp8_per_token(
            &quantized,
            &scales,
            num_tokens,
            head_dim,
            numr::dtype::DType::F32,
        )
        .unwrap();
    let cpu_deq_f32_vec = cpu_deq_f32.to_vec::<f32>();

    // FP8 e4m3 carries 3 mantissa bits, so FP8 quantization error dominates
    // over F16/BF16 output rounding.
    let rtol = 0.1;
    let atol = 0.01;

    // The CPU path narrows through numr's cast, which only handles F16/BF16
    // when boostr is built with the `f16` feature. CUDA below needs no feature.
    #[cfg(feature = "f16")]
    {
        for (dtype, label) in [
            (numr::dtype::DType::F16, "F16"),
            (numr::dtype::DType::BF16, "BF16"),
        ] {
            let narrow = cpu_client
                .dequantize_kv_fp8_per_token(&quantized, &scales, num_tokens, head_dim, dtype)
                .unwrap();
            assert_eq!(narrow.dtype(), dtype);
            let narrow_vec = narrow
                .to_dtype(numr::dtype::DType::F32)
                .unwrap()
                .to_vec::<f32>();
            assert_parity_f32_tol(
                &narrow_vec,
                &cpu_deq_f32_vec,
                &format!("dequantize_kv_fp8_per_token {label} output vs F32 output (CPU)"),
                rtol,
                atol,
            );
        }
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let inp = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        )
        .unwrap();
        let (cuda_quantized, cuda_scales) = cuda_client
            .quantize_kv_fp8_per_token(&inp, num_tokens, head_dim)
            .expect("quantize_kv_fp8_per_token must succeed on CUDA");

        let cuda_deq_f16 = cuda_client
            .dequantize_kv_fp8_per_token(
                &cuda_quantized,
                &cuda_scales,
                num_tokens,
                head_dim,
                DType::F16,
            )
            .expect("F16-output dequantize_kv_fp8_per_token must succeed on CUDA");
        assert_eq!(cuda_deq_f16.dtype(), DType::F16);
        let cuda_deq_f16_vec = cuda_deq_f16
            .to_dtype(DType::F32)
            .expect("cast F16 output to F32 for comparison")
            .to_vec::<f32>();
        assert_parity_f32_tol(
            &cuda_deq_f16_vec,
            &cpu_deq_f32_vec,
            "dequantize_kv_fp8_per_token F16 output CUDA vs CPU F32 reference",
            rtol,
            atol,
        );

        let cuda_deq_bf16 = cuda_client
            .dequantize_kv_fp8_per_token(
                &cuda_quantized,
                &cuda_scales,
                num_tokens,
                head_dim,
                DType::BF16,
            )
            .expect("BF16-output dequantize_kv_fp8_per_token must succeed on CUDA");
        assert_eq!(cuda_deq_bf16.dtype(), DType::BF16);
        let cuda_deq_bf16_vec = cuda_deq_bf16
            .to_dtype(DType::F32)
            .expect("cast BF16 output to F32 for comparison")
            .to_vec::<f32>();
        assert_parity_f32_tol(
            &cuda_deq_bf16_vec,
            &cpu_deq_f32_vec,
            "dequantize_kv_fp8_per_token BF16 output CUDA vs CPU F32 reference",
            rtol,
            atol,
        );
    });
}

// Covers dequantize_kv_int4's F16 and BF16 output-dtype kernels, which run on
// every supported device (they compile at sm_75, no capability gate needed).
#[test]
fn test_dequantize_kv_int4_narrow_output_dtypes() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let head_dim = 64;
    let group_size = Int4GroupSize::Group64;
    let input = det_tensor(&[num_tokens, head_dim], &cpu_device);

    let (packed, scales, zeros) = cpu_client
        .quantize_kv_int4(&input, num_tokens, head_dim, group_size)
        .unwrap();
    let cpu_deq_f32 = cpu_client
        .dequantize_kv_int4(
            &packed,
            &scales,
            &zeros,
            num_tokens,
            head_dim,
            group_size,
            numr::dtype::DType::F32,
        )
        .unwrap();
    let cpu_deq_f32_vec = cpu_deq_f32.to_vec::<f32>();

    // INT4 has 16 levels; this fixture spans about 1.0 per Group64 group, so
    // the quantization step is about 0.067. F16/BF16 output rounding is at
    // most ~4e-3 relative, far smaller than the INT4 step, so the step
    // dominates the tolerance rather than the output dtype.
    let rtol = 2e-2;
    let atol = 7e-2;

    // The CPU path narrows through numr's cast, which only handles F16/BF16
    // when boostr is built with the `f16` feature. CUDA below needs no feature.
    #[cfg(feature = "f16")]
    {
        for (dtype, label) in [
            (numr::dtype::DType::F16, "F16"),
            (numr::dtype::DType::BF16, "BF16"),
        ] {
            let narrow = cpu_client
                .dequantize_kv_int4(
                    &packed, &scales, &zeros, num_tokens, head_dim, group_size, dtype,
                )
                .unwrap();
            assert_eq!(narrow.dtype(), dtype);
            let narrow_vec = narrow
                .to_dtype(numr::dtype::DType::F32)
                .unwrap()
                .to_vec::<f32>();
            assert_parity_f32_tol(
                &narrow_vec,
                &cpu_deq_f32_vec,
                &format!("dequantize_kv_int4 {label} output vs F32 output (CPU)"),
                rtol,
                atol,
            );
        }
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let inp = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input.to_vec::<f32>(),
            &[num_tokens, head_dim],
            &cuda_device,
        )
        .unwrap();
        let (cuda_packed, cuda_scales, cuda_zeros) = cuda_client
            .quantize_kv_int4(&inp, num_tokens, head_dim, group_size)
            .expect("F32 quantize_kv_int4 must succeed on CUDA");

        let cuda_deq_f16 = cuda_client
            .dequantize_kv_int4(
                &cuda_packed,
                &cuda_scales,
                &cuda_zeros,
                num_tokens,
                head_dim,
                group_size,
                DType::F16,
            )
            .expect("F16-output dequantize_kv_int4 must succeed on CUDA");
        assert_eq!(cuda_deq_f16.dtype(), DType::F16);
        let cuda_deq_f16_vec = cuda_deq_f16
            .to_dtype(DType::F32)
            .expect("cast F16 output to F32 for comparison")
            .to_vec::<f32>();
        assert_parity_f32_tol(
            &cuda_deq_f16_vec,
            &cpu_deq_f32_vec,
            "dequantize_kv_int4 F16 output CUDA vs CPU F32 reference",
            rtol,
            atol,
        );

        let cuda_deq_bf16 = cuda_client
            .dequantize_kv_int4(
                &cuda_packed,
                &cuda_scales,
                &cuda_zeros,
                num_tokens,
                head_dim,
                group_size,
                DType::BF16,
            )
            .expect("BF16-output dequantize_kv_int4 must succeed on CUDA");
        assert_eq!(cuda_deq_bf16.dtype(), DType::BF16);
        let cuda_deq_bf16_vec = cuda_deq_bf16
            .to_dtype(DType::F32)
            .expect("cast BF16 output to F32 for comparison")
            .to_vec::<f32>();
        assert_parity_f32_tol(
            &cuda_deq_bf16_vec,
            &cpu_deq_f32_vec,
            "dequantize_kv_int4 BF16 output CUDA vs CPU F32 reference",
            rtol,
            atol,
        );
    });
}

// Covers `KvCacheOps::append_kv_int4`: quantizing one token at a time into a
// preallocated cache must agree with quantizing all tokens in one batch call,
// since both use the identical per-group min/max formula.
//
// Needs the `f16` feature. `append_kv_int4` stores scales and zeros as F16,
// while `dequantize_kv_int4` reads them as F32, so checking the round trip
// converts between the two. The CPU backend only casts F16 under that feature.
#[cfg(feature = "f16")]
#[test]
fn test_append_kv_int4_matches_batch_quantize() {
    use boostr::ops::traits::cache::kv_cache::KvCacheOps;
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let batch = 1;
    let heads = 1;
    let head_dim = 64;
    let max_seq_len = 4;
    let group_size = Int4GroupSize::Group64;
    let groups_per_token = 1; // head_dim / group_size

    let tokens_k: Vec<Vec<f32>> = (0..max_seq_len)
        .map(|t| {
            (0..head_dim)
                .map(|d| ((t * head_dim + d) as f32 * 0.037).sin())
                .collect()
        })
        .collect();
    let tokens_v: Vec<Vec<f32>> = (0..max_seq_len)
        .map(|t| {
            (0..head_dim)
                .map(|d| ((t * head_dim + d) as f32 * 0.071 + 1.0).cos())
                .collect()
        })
        .collect();

    let k_cache = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
        &[batch, heads, max_seq_len, head_dim / 2],
        DType::U8,
        &cpu_device,
    )
    .unwrap();
    let v_cache = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
        &[batch, heads, max_seq_len, head_dim / 2],
        DType::U8,
        &cpu_device,
    )
    .unwrap();
    let k_scales = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
        &[batch, heads, max_seq_len * groups_per_token],
        DType::F16,
        &cpu_device,
    )
    .unwrap();
    let k_zeros = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
        &[batch, heads, max_seq_len * groups_per_token],
        DType::F16,
        &cpu_device,
    )
    .unwrap();
    let v_scales = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
        &[batch, heads, max_seq_len * groups_per_token],
        DType::F16,
        &cpu_device,
    )
    .unwrap();
    let v_zeros = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
        &[batch, heads, max_seq_len * groups_per_token],
        DType::F16,
        &cpu_device,
    )
    .unwrap();

    for (t, (k_tok, v_tok)) in tokens_k.iter().zip(tokens_v.iter()).enumerate() {
        let new_k = Tensor::from_slice(k_tok, &[batch, heads, head_dim], &cpu_device).unwrap();
        let new_v = Tensor::from_slice(v_tok, &[batch, heads, head_dim], &cpu_device).unwrap();
        cpu_client
            .append_kv_int4(
                &k_cache, &v_cache, &k_scales, &k_zeros, &v_scales, &v_zeros, &new_k, &new_v, t,
                group_size,
            )
            .expect("append_kv_int4 must succeed on CPU");
    }

    // Reinterpret the [batch, heads, max_seq_len, head_dim/2] cache as the
    // flat [num_tokens, head_dim/2] shape `dequantize_kv_int4` expects.
    let k_packed_flat = Tensor::from_slice(
        &k_cache.to_vec::<u8>(),
        &[max_seq_len, head_dim / 2],
        &cpu_device,
    )
    .unwrap();
    let v_packed_flat = Tensor::from_slice(
        &v_cache.to_vec::<u8>(),
        &[max_seq_len, head_dim / 2],
        &cpu_device,
    )
    .unwrap();
    let k_scales_flat = Tensor::from_slice(
        &k_scales.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
        &[max_seq_len * groups_per_token],
        &cpu_device,
    )
    .unwrap();
    let k_zeros_flat = Tensor::from_slice(
        &k_zeros.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
        &[max_seq_len * groups_per_token],
        &cpu_device,
    )
    .unwrap();
    let v_scales_flat = Tensor::from_slice(
        &v_scales.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
        &[max_seq_len * groups_per_token],
        &cpu_device,
    )
    .unwrap();
    let v_zeros_flat = Tensor::from_slice(
        &v_zeros.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
        &[max_seq_len * groups_per_token],
        &cpu_device,
    )
    .unwrap();

    let k_deq_appended = cpu_client
        .dequantize_kv_int4(
            &k_packed_flat,
            &k_scales_flat,
            &k_zeros_flat,
            max_seq_len,
            head_dim,
            group_size,
            DType::F32,
        )
        .unwrap();
    let v_deq_appended = cpu_client
        .dequantize_kv_int4(
            &v_packed_flat,
            &v_scales_flat,
            &v_zeros_flat,
            max_seq_len,
            head_dim,
            group_size,
            DType::F32,
        )
        .unwrap();

    // Reference: quantize all tokens in one batch call. head_dim == group_size,
    // so `quantize_kv_int4`'s flat grouping lines up exactly with append's
    // per-token grouping.
    let k_all: Vec<f32> = tokens_k.iter().flatten().copied().collect();
    let v_all: Vec<f32> = tokens_v.iter().flatten().copied().collect();
    let k_input = Tensor::from_slice(&k_all, &[max_seq_len, head_dim], &cpu_device).unwrap();
    let v_input = Tensor::from_slice(&v_all, &[max_seq_len, head_dim], &cpu_device).unwrap();

    let (k_packed_b, k_scales_b, k_zeros_b) = cpu_client
        .quantize_kv_int4(&k_input, max_seq_len, head_dim, group_size)
        .unwrap();
    let (v_packed_b, v_scales_b, v_zeros_b) = cpu_client
        .quantize_kv_int4(&v_input, max_seq_len, head_dim, group_size)
        .unwrap();
    let k_deq_batch = cpu_client
        .dequantize_kv_int4(
            &k_packed_b,
            &k_scales_b,
            &k_zeros_b,
            max_seq_len,
            head_dim,
            group_size,
            DType::F32,
        )
        .unwrap();
    let v_deq_batch = cpu_client
        .dequantize_kv_int4(
            &v_packed_b,
            &v_scales_b,
            &v_zeros_b,
            max_seq_len,
            head_dim,
            group_size,
            DType::F32,
        )
        .unwrap();

    // Append's scale/zero round through F16 storage (the kernel contract);
    // batch quantize keeps them in F32. That is the only source of drift.
    assert_parity_f32_tol(
        &k_deq_appended.to_vec::<f32>(),
        &k_deq_batch.to_vec::<f32>(),
        "int4 append vs batch quantize (K)",
        1e-3,
        5e-3,
    );
    assert_parity_f32_tol(
        &v_deq_appended.to_vec::<f32>(),
        &v_deq_batch.to_vec::<f32>(),
        "int4 append vs batch quantize (V)",
        1e-3,
        5e-3,
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache::KvCacheOps as _;
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let cuda_k_cache = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
            &[batch, heads, max_seq_len, head_dim / 2],
            DType::U8,
            &cuda_device,
        )
        .unwrap();
        let cuda_v_cache = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
            &[batch, heads, max_seq_len, head_dim / 2],
            DType::U8,
            &cuda_device,
        )
        .unwrap();
        let cuda_k_scales = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
            &[batch, heads, max_seq_len * groups_per_token],
            DType::F16,
            &cuda_device,
        )
        .unwrap();
        let cuda_k_zeros = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
            &[batch, heads, max_seq_len * groups_per_token],
            DType::F16,
            &cuda_device,
        )
        .unwrap();
        let cuda_v_scales = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
            &[batch, heads, max_seq_len * groups_per_token],
            DType::F16,
            &cuda_device,
        )
        .unwrap();
        let cuda_v_zeros = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
            &[batch, heads, max_seq_len * groups_per_token],
            DType::F16,
            &cuda_device,
        )
        .unwrap();

        for (t, (k_tok, v_tok)) in tokens_k.iter().zip(tokens_v.iter()).enumerate() {
            let new_k = Tensor::from_slice(k_tok, &[batch, heads, head_dim], &cuda_device).unwrap();
            let new_v = Tensor::from_slice(v_tok, &[batch, heads, head_dim], &cuda_device).unwrap();
            cuda_client
                .append_kv_int4(
                    &cuda_k_cache,
                    &cuda_v_cache,
                    &cuda_k_scales,
                    &cuda_k_zeros,
                    &cuda_v_scales,
                    &cuda_v_zeros,
                    &new_k,
                    &new_v,
                    t,
                    group_size,
                )
                .expect("append_kv_int4 must succeed on CUDA");
        }

        let cuda_k_packed_flat = Tensor::from_slice(
            &cuda_k_cache.to_vec::<u8>(),
            &[max_seq_len, head_dim / 2],
            &cuda_device,
        )
        .unwrap();
        let cuda_k_scales_flat = Tensor::from_slice(
            &cuda_k_scales.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
            &[max_seq_len * groups_per_token],
            &cuda_device,
        )
        .unwrap();
        let cuda_k_zeros_flat = Tensor::from_slice(
            &cuda_k_zeros.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
            &[max_seq_len * groups_per_token],
            &cuda_device,
        )
        .unwrap();

        let cuda_k_deq = cuda_client
            .dequantize_kv_int4(
                &cuda_k_packed_flat,
                &cuda_k_scales_flat,
                &cuda_k_zeros_flat,
                max_seq_len,
                head_dim,
                group_size,
                DType::F32,
            )
            .unwrap();

        // Same formula, same F16 scale/zero storage on both sides: tolerance
        // only needs to absorb float rounding, not a full quant bucket.
        assert_parity_f32_tol(
            &cuda_k_deq.to_vec::<f32>(),
            &k_deq_appended.to_vec::<f32>(),
            "append_kv_int4 CUDA vs CPU reference (K)",
            1e-3,
            5e-3,
        );
    });
}
