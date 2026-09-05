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

// Covers quantize_kv_fp8_per_tensor/dequantize_kv_fp8_per_tensor: a single
// tensor-wide scale instead of one scale per token or per head.
#[test]
fn test_quantize_dequantize_fp8_per_tensor_roundtrip_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let shape = [4usize, 32usize];
    let input = det_tensor(&shape, &cpu_device);
    let orig = input.to_vec::<f32>();

    let (cpu_q, cpu_s) = cpu_client.quantize_kv_fp8_per_tensor(&input).unwrap();
    assert_eq!(cpu_q.dtype(), numr::dtype::DType::FP8E4M3);
    assert_eq!(cpu_s.shape(), &[1]);
    let cpu_deq = cpu_client
        .dequantize_kv_fp8_per_tensor(&cpu_q, &cpu_s, numr::dtype::DType::F32)
        .unwrap();
    let cpu_deq_vec = cpu_deq.to_vec::<f32>();

    // FP8 e4m3 keeps 3 mantissa bits, a relative step of ~1/8 near any
    // representable value; 15% covers that plus scale-rounding slop. The
    // error is relative (with an absolute floor for values near zero), not
    // a large flat absolute tolerance.
    let max_rel_err: f32 = orig
        .iter()
        .zip(cpu_deq_vec.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel_err < 0.15,
        "CPU FP8 per-tensor roundtrip relative error too high: {max_rel_err}"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        // CUDA's per-tensor kernel is F16-input/output only; build F32 then
        // narrow (`half::f16` is not a numr `Element`, so `from_slice`
        // cannot build an F16 tensor directly).
        let inp_f32 =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&orig, &shape, &cuda_device)
                .unwrap();
        let inp_f16 = inp_f32.to_dtype(DType::F16).expect("cast fixture to F16");

        let (q, s) = cuda_client
            .quantize_kv_fp8_per_tensor(&inp_f16)
            .expect("quantize_kv_fp8_per_tensor must succeed on CUDA");
        assert_eq!(q.dtype(), DType::FP8E4M3);
        assert_eq!(s.shape(), &[1]);
        let deq = cuda_client
            .dequantize_kv_fp8_per_tensor(&q, &s, DType::F16)
            .expect("dequantize_kv_fp8_per_tensor must succeed on CUDA")
            .to_dtype(DType::F32)
            .expect("cast F16 output to F32 for comparison");

        assert_parity_f32_tol(
            &deq.to_vec::<f32>(),
            &cpu_deq_vec,
            "fp8 per-tensor roundtrip CUDA vs CPU",
            0.1,  // 10% relative — FP8 has only 3 mantissa bits
            0.01, // absolute tolerance for values near zero
        );
    });
}

// Regression test for the multi-block per-tensor quantize bug: the original
// kernel wrote `scale` from block 0's local max only (`blockIdx.x == 0`),
// silently dropping every other block's contribution. The CUDA launcher's
// block size is 256 threads, so 1200 elements (not a multiple of 256) force
// 5 blocks (`ceil(1200 / 256)`) with a ragged last block, and the tensor is
// built so its max-abs element sits outside block 0 — a build that used the
// buggy single-block reduction would compute a scale from the wrong (too
// small) max and this test would fail.
#[test]
fn test_quantize_kv_fp8_per_tensor_multi_block() {
    let shape = [4usize, 300usize]; // 1200 elements, 5 blocks of 256
    let mut data: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.037).sin() * 0.4).collect();
    // Force the global max into the last (ragged) block, which the buggy
    // block-0-only implementation would never see.
    let last = data.len() - 1;
    data[last] = 6.0;

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let inp_f32 =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&data, &shape, &cuda_device)
                .unwrap();
        let inp_f16 = inp_f32.to_dtype(DType::F16).expect("cast fixture to F16");
        // Expected max-abs at F16 precision, matching what the kernel
        // actually reduces over.
        let f16_vals = inp_f16
            .to_dtype(DType::F32)
            .expect("cast back for reference")
            .to_vec::<f32>();
        let expected_max_abs = f16_vals.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let expected_scale = 448.0f32 / expected_max_abs;

        let (q, s) = cuda_client
            .quantize_kv_fp8_per_tensor(&inp_f16)
            .expect("quantize_kv_fp8_per_tensor must succeed on CUDA");
        let scale_val = s.to_vec::<f32>()[0];
        assert_parity_f32_tol(
            &[scale_val],
            &[expected_scale],
            "fp8 per-tensor multi-block scale (whole-tensor max, not block 0's)",
            1e-3,
            1e-6,
        );

        let deq = cuda_client
            .dequantize_kv_fp8_per_tensor(&q, &s, DType::F16)
            .expect("dequantize_kv_fp8_per_tensor must succeed on CUDA")
            .to_dtype(DType::F32)
            .expect("cast F16 output to F32 for comparison");
        let deq_vec = deq.to_vec::<f32>();

        let max_rel_err: f32 = f16_vals
            .iter()
            .zip(deq_vec.iter())
            .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        assert!(
            max_rel_err < 0.2,
            "fp8 per-tensor multi-block roundtrip relative error too high: {max_rel_err}"
        );
    });
    #[cfg(not(feature = "cuda"))]
    let _ = (&shape, &data);
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

// ---------------------------------------------------------------------------
// kv_fp8_bwd_per_{tensor,token}: backward of FP8 KV-cache fake-quantization.
//
// Synthetic fixtures are exact here: the backward takes (grad_output,
// kv_fp8, scale) directly, with no forward op needed to construct them.
//
// Both grad_kv AND grad_scale(s) are checked. grad_kv is a straight-through
// identity, so checking it alone would pass trivially and never exercise the
// block reduction that produces the scale gradient.
//
// head_dim (37) is not a power of two, and the per-tensor element count
// (300) does not divide FP8_BWD_BLOCK (256): both push the kernels' strided
// tree reduction through its partial (non-full-warp) path.
// ---------------------------------------------------------------------------

const FP8BWD_TOTAL_ELEMENTS: usize = 300;
const FP8BWD_BATCH: usize = 2;
const FP8BWD_KV_HEADS: usize = 3;
const FP8BWD_SEQ_LEN: usize = 5;
const FP8BWD_HEAD_DIM: usize = 37;
const FP8BWD_TOTAL_TOKENS: usize = FP8BWD_BATCH * FP8BWD_KV_HEADS * FP8BWD_SEQ_LEN;

/// Deterministic values in roughly [-2.0, 2.0], wide enough that FP8E4M3
/// encodes a spread of distinct codes rather than clustering near zero.
fn fp8bwd_vals(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.29 + phase).sin() * 2.0)
        .collect()
}

/// A plausible stored scale: the repo convention is `448 / max_abs`, so with
/// fixture values in [-2, 2] a realistic scale is on the order of 200, not
/// `max_abs / 448`.
const FP8BWD_SCALE: f32 = 200.0;

/// Tolerance scaled by the reference gradient's own magnitude, so a
/// small-valued gradient is not waved through by a fixed absolute tolerance.
#[cfg(feature = "cuda")]
fn assert_fp8bwd_grad_close(actual: &[f32], expected: &[f32], op: &str, rtol: f32, atol_frac: f32) {
    let max_abs = expected.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(max_abs > 0.0, "{op}: reference gradient is all zeros");
    assert_parity_f32_tol(actual, expected, op, rtol, atol_frac * max_abs);
}

#[test]
fn test_kv_fp8_bwd_per_tensor_cpu_reference_is_sane() {
    let (cpu_client, cpu_device) = setup_cpu();
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    let go_data = fp8bwd_vals(FP8BWD_TOTAL_ELEMENTS, 0.13);
    let kv_data = fp8bwd_vals(FP8BWD_TOTAL_ELEMENTS, 1.7);

    let go = Tensor::from_slice(&go_data, &[FP8BWD_TOTAL_ELEMENTS], &cpu_device).unwrap();
    let kv_f32 = Tensor::from_slice(&kv_data, &[FP8BWD_TOTAL_ELEMENTS], &cpu_device).unwrap();
    let kv_fp8 = kv_f32
        .to_dtype(DType::FP8E4M3)
        .expect("cast fixture to FP8E4M3");

    let (grad_kv, grad_scale) = cpu_client
        .kv_fp8_bwd_per_tensor(&go, &kv_fp8, FP8BWD_SCALE)
        .expect("CPU kv_fp8_bwd_per_tensor must succeed");

    assert_eq!(grad_kv.shape(), &[FP8BWD_TOTAL_ELEMENTS]);
    assert_eq!(grad_scale.shape(), &[1]);
    assert_eq!(
        grad_kv.to_vec::<f32>(),
        go_data,
        "grad_kv must equal grad_output (STE identity)"
    );
    let gs = grad_scale.to_vec::<f32>()[0];
    assert!(
        gs.is_finite() && gs != 0.0,
        "grad_scale must be a finite, non-zero reduction"
    );
}

#[cfg(feature = "cuda")]
fn run_kv_fp8_bwd_per_tensor_cuda_case(
    dtype: numr::dtype::DType,
    label: &str,
    rtol: f32,
    atol_frac: f32,
) {
    use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let go_data = fp8bwd_vals(FP8BWD_TOTAL_ELEMENTS, 0.13);
    let kv_data = fp8bwd_vals(FP8BWD_TOTAL_ELEMENTS, 1.7);

    let go_cpu = Tensor::from_slice(&go_data, &[FP8BWD_TOTAL_ELEMENTS], &cpu_device).unwrap();
    let kv_cpu_f32 = Tensor::from_slice(&kv_data, &[FP8BWD_TOTAL_ELEMENTS], &cpu_device).unwrap();
    let kv_cpu_fp8 = kv_cpu_f32
        .to_dtype(DType::FP8E4M3)
        .expect("cast CPU fixture to FP8E4M3");
    let (cpu_grad_kv, cpu_grad_scale) = cpu_client
        .kv_fp8_bwd_per_tensor(&go_cpu, &kv_cpu_fp8, FP8BWD_SCALE)
        .expect("CPU kv_fp8_bwd_per_tensor must succeed");
    let cpu_grad_kv_vec = cpu_grad_kv.to_vec::<f32>();
    let cpu_grad_scale_vec = cpu_grad_scale.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        let go_f32 = Tensor::from_slice(&go_data, &[FP8BWD_TOTAL_ELEMENTS], &cuda_device).unwrap();
        let go = if dtype == DType::F32 {
            go_f32
        } else {
            go_f32
                .to_dtype(dtype)
                .unwrap_or_else(|e| panic!("cast grad_output fixture to {dtype:?}: {e:?}"))
        };
        let kv_f32 = Tensor::from_slice(&kv_data, &[FP8BWD_TOTAL_ELEMENTS], &cuda_device).unwrap();
        let kv_fp8 = kv_f32
            .to_dtype(DType::FP8E4M3)
            .expect("cast CUDA fixture to FP8E4M3");

        let (grad_kv, grad_scale) = cuda_client
            .kv_fp8_bwd_per_tensor(&go, &kv_fp8, FP8BWD_SCALE)
            .unwrap_or_else(|e| panic!("{label} kv_fp8_bwd_per_tensor must succeed: {e:?}"));

        let grad_kv_f32 = if dtype == DType::F32 {
            grad_kv.to_vec::<f32>()
        } else {
            grad_kv
                .to_dtype(DType::F32)
                .expect("cast grad_kv result back to F32 for comparison")
                .to_vec::<f32>()
        };

        assert_fp8bwd_grad_close(
            &grad_kv_f32,
            &cpu_grad_kv_vec,
            &format!("kv_fp8_bwd_per_tensor grad_kv {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
        assert_fp8bwd_grad_close(
            &grad_scale.to_vec::<f32>(),
            &cpu_grad_scale_vec,
            &format!("kv_fp8_bwd_per_tensor grad_scale {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
    });
}

#[test]
fn test_kv_fp8_bwd_per_tensor_f32_cuda() {
    #[cfg(feature = "cuda")]
    run_kv_fp8_bwd_per_tensor_cuda_case(numr::dtype::DType::F32, "F32", 1e-4, 1e-5);
}

#[test]
fn test_kv_fp8_bwd_per_tensor_f16_cuda() {
    #[cfg(feature = "cuda")]
    run_kv_fp8_bwd_per_tensor_cuda_case(numr::dtype::DType::F16, "F16", 4e-2, 2e-2);
}

#[test]
fn test_kv_fp8_bwd_per_tensor_bf16_cuda() {
    #[cfg(feature = "cuda")]
    run_kv_fp8_bwd_per_tensor_cuda_case(numr::dtype::DType::BF16, "BF16", 4e-2, 2e-2);
}

#[test]
fn test_kv_fp8_bwd_per_token_cpu_reference_is_sane() {
    let (cpu_client, cpu_device) = setup_cpu();
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    let total = FP8BWD_TOTAL_TOKENS * FP8BWD_HEAD_DIM;
    let go_data = fp8bwd_vals(total, 0.31);
    let kv_data = fp8bwd_vals(total, 2.3);
    let scale_data: Vec<f32> = (0..FP8BWD_TOTAL_TOKENS)
        .map(|t| FP8BWD_SCALE + (t as f32))
        .collect();

    let go = Tensor::from_slice(
        &go_data,
        &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM],
        &cpu_device,
    )
    .unwrap();
    let kv_f32 = Tensor::from_slice(
        &kv_data,
        &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM],
        &cpu_device,
    )
    .unwrap();
    let kv_fp8 = kv_f32
        .to_dtype(DType::FP8E4M3)
        .expect("cast fixture to FP8E4M3");
    let scales = Tensor::from_slice(&scale_data, &[FP8BWD_TOTAL_TOKENS], &cpu_device).unwrap();

    let (grad_kv, grad_scales) = cpu_client
        .kv_fp8_bwd_per_token(
            &go,
            &kv_fp8,
            &scales,
            FP8BWD_BATCH,
            FP8BWD_KV_HEADS,
            FP8BWD_SEQ_LEN,
            FP8BWD_HEAD_DIM,
        )
        .expect("CPU kv_fp8_bwd_per_token must succeed");

    assert_eq!(grad_kv.shape(), &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM]);
    assert_eq!(grad_scales.shape(), &[FP8BWD_TOTAL_TOKENS]);
    assert_eq!(
        grad_kv.to_vec::<f32>(),
        go_data,
        "grad_kv must equal grad_output (STE identity)"
    );
    let gs = grad_scales.to_vec::<f32>();
    assert!(
        gs.iter().all(|x| x.is_finite()),
        "grad_scales must all be finite"
    );
    assert!(
        gs.iter().any(|x| *x != 0.0),
        "grad_scales must not be entirely zero"
    );
}

#[cfg(feature = "cuda")]
fn run_kv_fp8_bwd_per_token_cuda_case(
    dtype: numr::dtype::DType,
    label: &str,
    rtol: f32,
    atol_frac: f32,
) {
    use boostr::ops::traits::cache::kv_cache_quant::KvCacheQuantOps as _;
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let total = FP8BWD_TOTAL_TOKENS * FP8BWD_HEAD_DIM;
    let go_data = fp8bwd_vals(total, 0.31);
    let kv_data = fp8bwd_vals(total, 2.3);
    let scale_data: Vec<f32> = (0..FP8BWD_TOTAL_TOKENS)
        .map(|t| FP8BWD_SCALE + (t as f32))
        .collect();

    let go_cpu = Tensor::from_slice(
        &go_data,
        &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM],
        &cpu_device,
    )
    .unwrap();
    let kv_cpu_f32 = Tensor::from_slice(
        &kv_data,
        &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM],
        &cpu_device,
    )
    .unwrap();
    let kv_cpu_fp8 = kv_cpu_f32
        .to_dtype(DType::FP8E4M3)
        .expect("cast CPU fixture to FP8E4M3");
    let scales_cpu = Tensor::from_slice(&scale_data, &[FP8BWD_TOTAL_TOKENS], &cpu_device).unwrap();

    let (cpu_grad_kv, cpu_grad_scales) = cpu_client
        .kv_fp8_bwd_per_token(
            &go_cpu,
            &kv_cpu_fp8,
            &scales_cpu,
            FP8BWD_BATCH,
            FP8BWD_KV_HEADS,
            FP8BWD_SEQ_LEN,
            FP8BWD_HEAD_DIM,
        )
        .expect("CPU kv_fp8_bwd_per_token must succeed");
    let cpu_grad_kv_vec = cpu_grad_kv.to_vec::<f32>();
    let cpu_grad_scales_vec = cpu_grad_scales.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        let go_f32 = Tensor::from_slice(
            &go_data,
            &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM],
            &cuda_device,
        )
        .unwrap();
        let go = if dtype == DType::F32 {
            go_f32
        } else {
            go_f32
                .to_dtype(dtype)
                .unwrap_or_else(|e| panic!("cast grad_output fixture to {dtype:?}: {e:?}"))
        };
        let kv_f32 = Tensor::from_slice(
            &kv_data,
            &[FP8BWD_TOTAL_TOKENS, FP8BWD_HEAD_DIM],
            &cuda_device,
        )
        .unwrap();
        let kv_fp8 = kv_f32
            .to_dtype(DType::FP8E4M3)
            .expect("cast CUDA fixture to FP8E4M3");
        let scales = Tensor::from_slice(&scale_data, &[FP8BWD_TOTAL_TOKENS], &cuda_device).unwrap();

        let (grad_kv, grad_scales) = cuda_client
            .kv_fp8_bwd_per_token(
                &go,
                &kv_fp8,
                &scales,
                FP8BWD_BATCH,
                FP8BWD_KV_HEADS,
                FP8BWD_SEQ_LEN,
                FP8BWD_HEAD_DIM,
            )
            .unwrap_or_else(|e| panic!("{label} kv_fp8_bwd_per_token must succeed: {e:?}"));

        let grad_kv_f32 = if dtype == DType::F32 {
            grad_kv.to_vec::<f32>()
        } else {
            grad_kv
                .to_dtype(DType::F32)
                .expect("cast grad_kv result back to F32 for comparison")
                .to_vec::<f32>()
        };

        assert_fp8bwd_grad_close(
            &grad_kv_f32,
            &cpu_grad_kv_vec,
            &format!("kv_fp8_bwd_per_token grad_kv {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
        assert_fp8bwd_grad_close(
            &grad_scales.to_vec::<f32>(),
            &cpu_grad_scales_vec,
            &format!("kv_fp8_bwd_per_token grad_scales {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
    });
}

#[test]
fn test_kv_fp8_bwd_per_token_f32_cuda() {
    #[cfg(feature = "cuda")]
    run_kv_fp8_bwd_per_token_cuda_case(numr::dtype::DType::F32, "F32", 1e-4, 1e-5);
}

#[test]
fn test_kv_fp8_bwd_per_token_f16_cuda() {
    #[cfg(feature = "cuda")]
    run_kv_fp8_bwd_per_token_cuda_case(numr::dtype::DType::F16, "F16", 4e-2, 2e-2);
}

#[test]
fn test_kv_fp8_bwd_per_token_bf16_cuda() {
    #[cfg(feature = "cuda")]
    run_kv_fp8_bwd_per_token_cuda_case(numr::dtype::DType::BF16, "BF16", 4e-2, 2e-2);
}
