//! Backend parity tests for NF4 operations

use super::helpers::*;
use boostr::DequantOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[test]
fn test_nf4_dequant_codebook_values() {
    let (cpu_client, cpu_device) = setup_cpu();

    // Create data with known codebook indices: all zeros (idx 0 = 0.0)
    let nf4_bytes = vec![0u8; 16]; // 16 bytes = 32 elements, all index 0
    let nf4_data = Tensor::<CpuRuntime>::from_slice(&nf4_bytes, &[16], &cpu_device);
    let absmax = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &cpu_device);

    let result = cpu_client.nf4_dequant(&nf4_data, &absmax, 32).unwrap();
    assert_eq!(result.shape(), &[32]);

    let data = result.to_vec::<f32>();
    for &v in &data {
        assert!((v - 0.0).abs() < 1e-6, "expected 0.0, got {}", v);
    }
}

#[test]
fn test_nf4_dequant_scaling() {
    let (cpu_client, cpu_device) = setup_cpu();

    // Index 15 = codebook value 1.0, absmax = 2.0 → output = 2.0
    let nf4_bytes = vec![0xFFu8; 16]; // all index 15
    let nf4_data = Tensor::<CpuRuntime>::from_slice(&nf4_bytes, &[16], &cpu_device);
    let absmax = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &cpu_device);

    let result = cpu_client.nf4_dequant(&nf4_data, &absmax, 32).unwrap();
    let data = result.to_vec::<f32>();
    for &v in &data {
        assert!((v - 2.0).abs() < 1e-6, "expected 2.0, got {}", v);
    }
}

#[test]
fn test_nf4_gemm_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let m = 2;
    let k = 32;
    let n = 8;
    let blocksize = 32;

    let input = det_tensor(&[m, k], &cpu_device);

    // NF4 weight: [N*K/2] bytes
    let num_bytes = n * k / 2;
    let nf4_bytes: Vec<u8> = (0..num_bytes)
        .map(|i| {
            let lo = (i % 16) as u8;
            let hi = ((i + 5) % 16) as u8;
            (hi << 4) | lo
        })
        .collect();
    let nf4_weight = Tensor::<CpuRuntime>::from_slice(&nf4_bytes, &[num_bytes], &cpu_device);

    // Absmax: [N * K/blocksize]
    let num_absmax = n * k / blocksize;
    let absmax_data: Vec<f32> = (0..num_absmax).map(|i| 0.5 + i as f32 * 0.1).collect();
    let absmax = Tensor::<CpuRuntime>::from_slice(&absmax_data, &[num_absmax], &cpu_device);

    let cpu_result = cpu_client
        .nf4_gemm(&input, &nf4_weight, &absmax, n, k, blocksize)
        .unwrap();
    assert_eq!(cpu_result.shape(), &[m, n]);

    let cpu_vec = cpu_result.to_vec::<f32>();
    for &v in &cpu_vec {
        assert!(v.is_finite(), "non-finite value: {}", v);
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::DequantOps as _;
        use numr::tensor::Tensor;

        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &cuda_device);
        let nf4_c =
            Tensor::from_slice(&nf4_weight.to_vec::<u8>(), nf4_weight.shape(), &cuda_device);
        let absmax_c = Tensor::from_slice(&absmax.to_vec::<f32>(), absmax.shape(), &cuda_device);

        let cuda_result = cuda_client
            .nf4_gemm(&input_c, &nf4_c, &absmax_c, n, k, blocksize)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_result.to_vec::<f32>(),
            &cpu_vec,
            "nf4_gemm CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::DequantOps as _;
        use numr::tensor::Tensor;

        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &wgpu_device);
        let nf4_w =
            Tensor::from_slice(&nf4_weight.to_vec::<u8>(), nf4_weight.shape(), &wgpu_device);
        let absmax_w = Tensor::from_slice(&absmax.to_vec::<f32>(), absmax.shape(), &wgpu_device);

        let wgpu_result = wgpu_client
            .nf4_gemm(&input_w, &nf4_w, &absmax_w, n, k, blocksize)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_result.to_vec::<f32>(),
            &cpu_vec,
            "nf4_gemm WebGPU vs CPU",
        );
    });
}

#[test]
fn test_nf4_gemm_matches_dequant_matmul() {
    use numr::ops::MatmulOps;

    let (cpu_client, cpu_device) = setup_cpu();

    let m = 2;
    let k = 32;
    let n = 4;
    let blocksize = 32;

    let input = det_tensor(&[m, k], &cpu_device);

    // Create NF4 weight data
    let num_bytes = n * k / 2;
    let nf4_bytes: Vec<u8> = (0..num_bytes)
        .map(|i| {
            let lo = (i % 16) as u8;
            let hi = ((i + 7) % 16) as u8;
            (hi << 4) | lo
        })
        .collect();
    let nf4_weight = Tensor::<CpuRuntime>::from_slice(&nf4_bytes, &[num_bytes], &cpu_device);

    let num_absmax = n * k / blocksize;
    let absmax_data: Vec<f32> = (0..num_absmax).map(|i| 1.0 + i as f32 * 0.05).collect();
    let absmax = Tensor::<CpuRuntime>::from_slice(&absmax_data, &[num_absmax], &cpu_device);

    // Method 1: Fused NF4 GEMM
    let fused_result = cpu_client
        .nf4_gemm(&input, &nf4_weight, &absmax, n, k, blocksize)
        .unwrap();

    // Method 2: Dequant then matmul
    let dequant_weight = cpu_client
        .nf4_dequant(&nf4_weight, &absmax, blocksize)
        .unwrap();
    // Reshape to [N, K] and transpose to [K, N] for matmul
    let dequant_2d = dequant_weight.reshape(&[n, k]).unwrap();
    let dequant_t = dequant_2d.transpose(0isize, 1isize).unwrap().contiguous();
    let ref_result = MatmulOps::matmul(&cpu_client, &input, &dequant_t).unwrap();

    let cpu_fused_vec = fused_result.to_vec::<f32>();
    assert_parity_f32_relaxed(
        &cpu_fused_vec,
        &ref_result.to_vec::<f32>(),
        "nf4_gemm vs dequant+matmul",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::DequantOps as _;
        use numr::tensor::Tensor;

        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &cuda_device);
        let nf4_c =
            Tensor::from_slice(&nf4_weight.to_vec::<u8>(), nf4_weight.shape(), &cuda_device);
        let absmax_c = Tensor::from_slice(&absmax.to_vec::<f32>(), absmax.shape(), &cuda_device);

        let cuda_fused = cuda_client
            .nf4_gemm(&input_c, &nf4_c, &absmax_c, n, k, blocksize)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_fused.to_vec::<f32>(),
            &cpu_fused_vec,
            "nf4_gemm fused CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::DequantOps as _;
        use numr::tensor::Tensor;

        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &wgpu_device);
        let nf4_w =
            Tensor::from_slice(&nf4_weight.to_vec::<u8>(), nf4_weight.shape(), &wgpu_device);
        let absmax_w = Tensor::from_slice(&absmax.to_vec::<f32>(), absmax.shape(), &wgpu_device);

        let wgpu_fused = wgpu_client
            .nf4_gemm(&input_w, &nf4_w, &absmax_w, n, k, blocksize)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_fused.to_vec::<f32>(),
            &cpu_fused_vec,
            "nf4_gemm fused WebGPU vs CPU",
        );
    });
}
