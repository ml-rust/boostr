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
