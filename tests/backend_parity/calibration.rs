//! Backend parity tests for CalibrationOps.

use super::helpers::*;
use boostr::ops::traits::CalibrationOps;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

// ============================================================================
// AWQ channel scores
// ============================================================================

#[test]
fn test_awq_channel_scores_cpu() {
    let (client, device) = setup_cpu();
    let act = det_tensor(&[4, 8], &device);
    let w = det_tensor(&[6, 8], &device);

    let result = client.awq_channel_scores(&act, &w).unwrap();
    assert_eq!(result.shape(), &[8]);

    let data = result.to_vec::<f32>();
    for &v in &data {
        assert!(v >= 0.0, "scores should be non-negative, got {}", v);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_awq_channel_scores_cuda_parity() {
    let (_cpu_client, cpu_device) = setup_cpu();
    let act_data = det_tensor(&[4, 8], &cpu_device);
    let w_data = det_tensor(&[6, 8], &cpu_device);

    let cpu_result = _cpu_client.awq_channel_scores(&act_data, &w_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let act_c =
            Tensor::<CudaRuntime>::from_slice(&act_data.to_vec::<f32>(), &[4, 8], &cuda_device);
        let w_c = Tensor::<CudaRuntime>::from_slice(&w_data.to_vec::<f32>(), &[6, 8], &cuda_device);

        let result = cuda_client.awq_channel_scores(&act_c, &w_c).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "awq_channel_scores CUDA vs CPU",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_awq_channel_scores_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let act_data = det_tensor(&[4, 8], &cpu_device);
    let w_data = det_tensor(&[6, 8], &cpu_device);

    let cpu_result = cpu_client.awq_channel_scores(&act_data, &w_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let act_w =
            Tensor::<WgpuRuntime>::from_slice(&act_data.to_vec::<f32>(), &[4, 8], &wgpu_device);
        let w_w = Tensor::<WgpuRuntime>::from_slice(&w_data.to_vec::<f32>(), &[6, 8], &wgpu_device);

        let result = wgpu_client.awq_channel_scores(&act_w, &w_w).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "awq_channel_scores WGPU vs CPU",
        );
    });
}

// ============================================================================
// Fisher information
// ============================================================================

#[test]
fn test_fisher_information_cpu() {
    let (client, device) = setup_cpu();
    let grads = det_tensor(&[16, 32], &device);

    let result = client.fisher_information(&grads).unwrap();
    assert_eq!(result.shape(), &[32]);

    let data = result.to_vec::<f32>();
    for &v in &data {
        assert!(v >= 0.0, "Fisher values should be non-negative, got {}", v);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fisher_information_cuda_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let grad_data = det_tensor(&[16, 32], &cpu_device);

    let cpu_result = cpu_client.fisher_information(&grad_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let grad_c =
            Tensor::<CudaRuntime>::from_slice(&grad_data.to_vec::<f32>(), &[16, 32], &cuda_device);

        let result = cuda_client.fisher_information(&grad_c).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "fisher_information CUDA vs CPU",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fisher_information_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let grad_data = det_tensor(&[16, 32], &cpu_device);

    let cpu_result = cpu_client.fisher_information(&grad_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let grad_w =
            Tensor::<WgpuRuntime>::from_slice(&grad_data.to_vec::<f32>(), &[16, 32], &wgpu_device);

        let result = wgpu_client.fisher_information(&grad_w).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "fisher_information WGPU vs CPU",
        );
    });
}

// ============================================================================
// GPTQ Hessian update
// ============================================================================

#[test]
fn test_gptq_hessian_update_cpu() {
    let (client, device) = setup_cpu();
    let h = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &device);
    let x = det_tensor(&[4, 8], &device);

    let result = client.gptq_hessian_update(&h, &x).unwrap();
    assert_eq!(result.shape(), &[8, 8]);

    // Verify symmetry: H = X^T X should be symmetric
    let data = result.to_vec::<f32>();
    for i in 0..8 {
        for j in 0..8 {
            let diff = (data[i * 8 + j] - data[j * 8 + i]).abs();
            assert!(
                diff < 1e-5,
                "not symmetric at [{},{}]: {} vs {}",
                i,
                j,
                data[i * 8 + j],
                data[j * 8 + i]
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gptq_hessian_update_cuda_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let h_data = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &cpu_device);
    let x_data = det_tensor(&[4, 8], &cpu_device);

    let cpu_result = cpu_client.gptq_hessian_update(&h_data, &x_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let h_c = Tensor::<CudaRuntime>::from_slice(&h_data.to_vec::<f32>(), &[8, 8], &cuda_device);
        let x_c = Tensor::<CudaRuntime>::from_slice(&x_data.to_vec::<f32>(), &[4, 8], &cuda_device);

        let result = cuda_client.gptq_hessian_update(&h_c, &x_c).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "gptq_hessian_update CUDA vs CPU",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gptq_hessian_update_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let h_data = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &cpu_device);
    let x_data = det_tensor(&[4, 8], &cpu_device);

    let cpu_result = cpu_client.gptq_hessian_update(&h_data, &x_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let h_w = Tensor::<WgpuRuntime>::from_slice(&h_data.to_vec::<f32>(), &[8, 8], &wgpu_device);
        let x_w = Tensor::<WgpuRuntime>::from_slice(&x_data.to_vec::<f32>(), &[4, 8], &wgpu_device);

        let result = wgpu_client.gptq_hessian_update(&h_w, &x_w).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "gptq_hessian_update WGPU vs CPU",
        );
    });
}

// ============================================================================
// GPTQ column quantization
// ============================================================================

#[test]
fn test_gptq_quantize_column_cpu() {
    let (client, device) = setup_cpu();

    let w_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[8, 16], &device);

    // Identity H_inv
    let mut h_inv_data = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        h_inv_data[i * 16 + i] = 1.0;
    }
    let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[16, 16], &device);

    let (q, scales, zeros) = client
        .gptq_quantize_column(&w, &h_inv, 4, 4, false)
        .unwrap();

    assert_eq!(q.shape(), &[8, 16]);
    assert_eq!(scales.shape(), &[8, 4]);
    assert_eq!(zeros.shape(), &[8, 4]);

    // Verify shapes and that scales are positive
    let q_data = q.to_vec::<f32>();
    let s_data = scales.to_vec::<f32>();
    assert_eq!(q_data.len(), 128);
    for &s in &s_data {
        assert!(s > 0.0, "scale should be positive, got {}", s);
    }
    // Quantized values should be finite
    for (i, &v) in q_data.iter().enumerate() {
        assert!(v.is_finite(), "non-finite quantized value at {}: {}", i, v);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gptq_quantize_column_cuda_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let w_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[8, 16], &cpu_device);

    let mut h_inv_data = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        h_inv_data[i * 16 + i] = 1.0;
    }
    let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[16, 16], &cpu_device);

    let (cpu_q, cpu_s, cpu_z) = cpu_client
        .gptq_quantize_column(&w, &h_inv, 4, 4, false)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_s_vec = cpu_s.to_vec::<f32>();
    let cpu_z_vec = cpu_z.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let w_c = Tensor::<CudaRuntime>::from_slice(&w_data, &[8, 16], &cuda_device);
        let h_c = Tensor::<CudaRuntime>::from_slice(&h_inv_data, &[16, 16], &cuda_device);

        let (q, s, z) = cuda_client
            .gptq_quantize_column(&w_c, &h_c, 4, 4, false)
            .unwrap();
        // Relaxed tolerance — accumulated quantization error across columns
        assert_parity_f32_relaxed(
            &q.to_vec::<f32>(),
            &cpu_q_vec,
            "gptq_quantize_column CUDA vs CPU (q)",
        );
        assert_parity_f32_relaxed(
            &s.to_vec::<f32>(),
            &cpu_s_vec,
            "gptq_quantize_column CUDA vs CPU (scales)",
        );
        assert_parity_f32_relaxed(
            &z.to_vec::<f32>(),
            &cpu_z_vec,
            "gptq_quantize_column CUDA vs CPU (zeros)",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gptq_quantize_column_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let w_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[8, 16], &cpu_device);

    let mut h_inv_data = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        h_inv_data[i * 16 + i] = 1.0;
    }
    let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[16, 16], &cpu_device);

    let (cpu_q, cpu_s, cpu_z) = cpu_client
        .gptq_quantize_column(&w, &h_inv, 4, 4, false)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_s_vec = cpu_s.to_vec::<f32>();
    let cpu_z_vec = cpu_z.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let w_w = Tensor::<WgpuRuntime>::from_slice(&w_data, &[8, 16], &wgpu_device);
        let h_w = Tensor::<WgpuRuntime>::from_slice(&h_inv_data, &[16, 16], &wgpu_device);

        let (q, s, z) = wgpu_client
            .gptq_quantize_column(&w_w, &h_w, 4, 4, false)
            .unwrap();
        assert_parity_f32_relaxed(
            &q.to_vec::<f32>(),
            &cpu_q_vec,
            "gptq_quantize_column WGPU vs CPU (q)",
        );
        assert_parity_f32_relaxed(
            &s.to_vec::<f32>(),
            &cpu_s_vec,
            "gptq_quantize_column WGPU vs CPU (scales)",
        );
        assert_parity_f32_relaxed(
            &z.to_vec::<f32>(),
            &cpu_z_vec,
            "gptq_quantize_column WGPU vs CPU (zeros)",
        );
    });
}
