//! Backend parity tests for INT4 GEMM operations (AWQ, GPTQ, Marlin)

use super::helpers::*;
use boostr::QuantMatmulOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Create AWQ-packed test data: qweight [K, N/8], scales [num_groups, N], zeros [num_groups, N]
fn create_awq_test_data(
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    device: &CpuDevice,
) -> (
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
) {
    // Input activation
    let input = det_tensor(&[m, k], device);

    // AWQ packed weights: [K, N/8] as u32
    // Pack 8 nibbles per u32 with AWQ shift pattern [0,16,4,20,8,24,12,28]
    let awq_shifts = [0u32, 16, 4, 20, 8, 24, 12, 28];
    let n_packed = n / 8;
    let mut qweight_data = vec![0u32; k * n_packed];
    for ki in 0..k {
        for pj in 0..n_packed {
            let mut packed = 0u32;
            for sub in 0..8 {
                // Use deterministic pattern: nibble values 0-15
                let val = ((ki * n_packed + pj + sub) % 16) as u32;
                packed |= val << awq_shifts[sub];
            }
            qweight_data[ki * n_packed + pj] = packed;
        }
    }
    let qweight = Tensor::<CpuRuntime>::from_slice(
        bytemuck::cast_slice::<u32, f32>(&qweight_data),
        &[k, n_packed],
        device,
    );

    // Scales and zeros: [num_groups, N]
    let num_groups = k / group_size;
    let scales_data: Vec<f32> = (0..num_groups * n)
        .map(|i| 0.01 + (i as f32 * 0.001).sin().abs() * 0.1)
        .collect();
    let zeros_data: Vec<f32> = (0..num_groups * n)
        .map(|i| 7.0 + (i as f32 * 0.003).cos() * 0.5)
        .collect();
    let scales = Tensor::<CpuRuntime>::from_slice(&scales_data, &[num_groups, n], device);
    let zeros = Tensor::<CpuRuntime>::from_slice(&zeros_data, &[num_groups, n], device);

    (input, qweight, scales, zeros)
}

/// Create GPTQ-packed test data
fn create_gptq_test_data(
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    device: &CpuDevice,
) -> (
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
) {
    let input = det_tensor(&[m, k], device);

    // GPTQ packed weights: [K/8, N] sequential 4-bit packing
    let k_packed = k / 8;
    let mut qweight_data = vec![0u32; k_packed * n];
    for pk in 0..k_packed {
        for col in 0..n {
            let mut packed = 0u32;
            for sub in 0..8 {
                let val = ((pk * 8 + sub + col) % 16) as u32;
                packed |= val << (sub * 4);
            }
            qweight_data[pk * n + col] = packed;
        }
    }
    let qweight = Tensor::<CpuRuntime>::from_slice(
        bytemuck::cast_slice::<u32, f32>(&qweight_data),
        &[k_packed, n],
        device,
    );

    // GPTQ qzeros: [num_groups, N/8] packed u32
    let num_groups = k / group_size;
    let n_packed_zeros = n / 8;
    let mut qzeros_data = vec![0u32; num_groups * n_packed_zeros];
    for g in 0..num_groups {
        for pn in 0..n_packed_zeros {
            let mut packed = 0u32;
            for sub in 0..8 {
                packed |= 8u32 << (sub * 4); // zero point = 8
            }
            qzeros_data[g * n_packed_zeros + pn] = packed;
        }
    }
    let qzeros = Tensor::<CpuRuntime>::from_slice(
        bytemuck::cast_slice::<u32, f32>(&qzeros_data),
        &[num_groups, n_packed_zeros],
        device,
    );

    // Scales: [num_groups, N]
    let scales_data: Vec<f32> = (0..num_groups * n)
        .map(|i| 0.01 + (i as f32 * 0.002).sin().abs() * 0.1)
        .collect();
    let scales = Tensor::<CpuRuntime>::from_slice(&scales_data, &[num_groups, n], device);

    // g_idx: [K] — simple sequential grouping
    let g_idx_data: Vec<i32> = (0..k).map(|i| (i / group_size) as i32).collect();
    let g_idx = Tensor::<CpuRuntime>::from_slice(
        bytemuck::cast_slice::<i32, f32>(&g_idx_data),
        &[k],
        device,
    );

    (input, qweight, qzeros, scales, g_idx)
}

/// Create Marlin-packed test data
fn create_marlin_test_data(
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    device: &CpuDevice,
) -> (
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
) {
    let input = det_tensor(&[m, k], device);

    // Marlin: [K/8, N] sequential 4-bit packing
    let k_packed = k / 8;
    let mut weight_data = vec![0u32; k_packed * n];
    for pk in 0..k_packed {
        for col in 0..n {
            let mut packed = 0u32;
            for sub in 0..8 {
                let val = ((pk * 8 + sub + col) % 16) as u32;
                packed |= val << (sub * 4);
            }
            weight_data[pk * n + col] = packed;
        }
    }
    let weight = Tensor::<CpuRuntime>::from_slice(
        bytemuck::cast_slice::<u32, f32>(&weight_data),
        &[k_packed, n],
        device,
    );

    let num_groups = k / group_size;
    let scales_data: Vec<f32> = (0..num_groups * n)
        .map(|i| 0.01 + (i as f32 * 0.001).sin().abs() * 0.1)
        .collect();
    let zeros_data: Vec<f32> = (0..num_groups * n)
        .map(|i| (i as f32 * 0.002).cos() * 0.01)
        .collect();
    let scales = Tensor::<CpuRuntime>::from_slice(&scales_data, &[num_groups, n], device);
    let zeros = Tensor::<CpuRuntime>::from_slice(&zeros_data, &[num_groups, n], device);

    (input, weight, scales, zeros)
}

#[test]
fn test_int4_gemm_awq_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (input, qweight, scales, zeros) = create_awq_test_data(2, 32, 16, 32, &cpu_device);

    let cpu_result = cpu_client
        .int4_gemm(&input, &qweight, &scales, &zeros, 32)
        .unwrap();
    assert_eq!(cpu_result.shape(), &[2, 16]);
    assert_eq!(cpu_result.dtype(), DType::F32);

    let cpu_vec = cpu_result.to_vec::<f32>();
    for &v in &cpu_vec {
        assert!(v.is_finite(), "non-finite value: {}", v);
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::QuantMatmulOps as _;
        use numr::tensor::Tensor;

        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &cuda_device);
        let qw_c = Tensor::from_slice(&qweight.to_vec::<f32>(), qweight.shape(), &cuda_device);
        let sc_c = Tensor::from_slice(&scales.to_vec::<f32>(), scales.shape(), &cuda_device);
        let zr_c = Tensor::from_slice(&zeros.to_vec::<f32>(), zeros.shape(), &cuda_device);

        let cuda_result = cuda_client
            .int4_gemm(&input_c, &qw_c, &sc_c, &zr_c, 32)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_result.to_vec::<f32>(),
            &cpu_vec,
            "int4_gemm AWQ CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::QuantMatmulOps as _;
        use numr::tensor::Tensor;

        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &wgpu_device);
        let qw_w = Tensor::from_slice(&qweight.to_vec::<f32>(), qweight.shape(), &wgpu_device);
        let sc_w = Tensor::from_slice(&scales.to_vec::<f32>(), scales.shape(), &wgpu_device);
        let zr_w = Tensor::from_slice(&zeros.to_vec::<f32>(), zeros.shape(), &wgpu_device);

        let wgpu_result = wgpu_client
            .int4_gemm(&input_w, &qw_w, &sc_w, &zr_w, 32)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_result.to_vec::<f32>(),
            &cpu_vec,
            "int4_gemm AWQ WebGPU vs CPU",
        );
    });
}

#[test]
fn test_int4_gemm_gptq_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (input, qweight, qzeros, scales, g_idx) = create_gptq_test_data(2, 32, 16, 32, &cpu_device);

    let cpu_result = cpu_client
        .int4_gemm_gptq(&input, &qweight, &qzeros, &scales, &g_idx)
        .unwrap();
    assert_eq!(cpu_result.shape(), &[2, 16]);

    let cpu_vec = cpu_result.to_vec::<f32>();
    for &v in &cpu_vec {
        assert!(v.is_finite(), "non-finite value: {}", v);
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::QuantMatmulOps as _;
        use numr::tensor::Tensor;

        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &cuda_device);
        let qw_c = Tensor::from_slice(&qweight.to_vec::<f32>(), qweight.shape(), &cuda_device);
        let qz_c = Tensor::from_slice(&qzeros.to_vec::<f32>(), qzeros.shape(), &cuda_device);
        let sc_c = Tensor::from_slice(&scales.to_vec::<f32>(), scales.shape(), &cuda_device);
        let gi_c = Tensor::from_slice(&g_idx.to_vec::<f32>(), g_idx.shape(), &cuda_device);

        let cuda_result = cuda_client
            .int4_gemm_gptq(&input_c, &qw_c, &qz_c, &sc_c, &gi_c)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_result.to_vec::<f32>(),
            &cpu_vec,
            "int4_gemm GPTQ CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::QuantMatmulOps as _;
        use numr::tensor::Tensor;

        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &wgpu_device);
        let qw_w = Tensor::from_slice(&qweight.to_vec::<f32>(), qweight.shape(), &wgpu_device);
        let qz_w = Tensor::from_slice(&qzeros.to_vec::<f32>(), qzeros.shape(), &wgpu_device);
        let sc_w = Tensor::from_slice(&scales.to_vec::<f32>(), scales.shape(), &wgpu_device);
        let gi_w = Tensor::from_slice(&g_idx.to_vec::<f32>(), g_idx.shape(), &wgpu_device);

        let wgpu_result = wgpu_client
            .int4_gemm_gptq(&input_w, &qw_w, &qz_w, &sc_w, &gi_w)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_result.to_vec::<f32>(),
            &cpu_vec,
            "int4_gemm GPTQ WebGPU vs CPU",
        );
    });
}

#[test]
fn test_marlin_gemm_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (input, weight, scales, zeros) = create_marlin_test_data(2, 32, 16, 32, &cpu_device);

    let cpu_result = cpu_client
        .marlin_gemm(&input, &weight, &scales, &zeros, 32)
        .unwrap();
    assert_eq!(cpu_result.shape(), &[2, 16]);

    let cpu_vec = cpu_result.to_vec::<f32>();
    for &v in &cpu_vec {
        assert!(v.is_finite(), "non-finite value: {}", v);
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::QuantMatmulOps as _;
        use numr::tensor::Tensor;

        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &cuda_device);
        let wt_c = Tensor::from_slice(&weight.to_vec::<f32>(), weight.shape(), &cuda_device);
        let sc_c = Tensor::from_slice(&scales.to_vec::<f32>(), scales.shape(), &cuda_device);
        let zr_c = Tensor::from_slice(&zeros.to_vec::<f32>(), zeros.shape(), &cuda_device);

        let cuda_result = cuda_client
            .marlin_gemm(&input_c, &wt_c, &sc_c, &zr_c, 32)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_result.to_vec::<f32>(),
            &cpu_vec,
            "marlin_gemm CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::QuantMatmulOps as _;
        use numr::tensor::Tensor;

        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), input.shape(), &wgpu_device);
        let wt_w = Tensor::from_slice(&weight.to_vec::<f32>(), weight.shape(), &wgpu_device);
        let sc_w = Tensor::from_slice(&scales.to_vec::<f32>(), scales.shape(), &wgpu_device);
        let zr_w = Tensor::from_slice(&zeros.to_vec::<f32>(), zeros.shape(), &wgpu_device);

        let wgpu_result = wgpu_client
            .marlin_gemm(&input_w, &wt_w, &sc_w, &zr_w, 32)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_result.to_vec::<f32>(),
            &cpu_vec,
            "marlin_gemm WebGPU vs CPU",
        );
    });
}
