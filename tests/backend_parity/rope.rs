//! Backend parity tests for RoPEOps.

use super::helpers::*;
use boostr::ops::traits::position::rope::RoPEOps;
use numr::autograd::Var;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn test_apply_rope_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 8, 32);

    let x_data = det_tensor(&[b, h, s, d], &cpu_device);
    let cos_data = det_tensor(&[s, d / 2], &cpu_device);
    let sin_data = det_tensor(&[s, d / 2], &cpu_device);

    let x = Var::<CpuRuntime>::new(x_data.clone(), false);
    let cos = Var::<CpuRuntime>::new(cos_data.clone(), false);
    let sin = Var::<CpuRuntime>::new(sin_data.clone(), false);

    let cpu_result = cpu_client.apply_rope(&x, &cos, &sin).unwrap();
    let cpu_result_vec = cpu_result.tensor().to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::rope::RoPEOps as _;
        use numr::autograd::Var;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;
        let x_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&x_data.to_vec::<f32>(), &[b, h, s, d], &cuda_device),
            false,
        );
        let cos_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[s, d / 2], &cuda_device),
            false,
        );
        let sin_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[s, d / 2], &cuda_device),
            false,
        );
        let result = cuda_client.apply_rope(&x_c, &cos_c, &sin_c).unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::position::rope::RoPEOps as _;
        use numr::autograd::Var;
        use numr::runtime::wgpu::WgpuRuntime;
        use numr::tensor::Tensor;
        let x_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&x_data.to_vec::<f32>(), &[b, h, s, d], &wgpu_device),
            false,
        );
        let cos_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[s, d / 2], &wgpu_device),
            false,
        );
        let sin_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[s, d / 2], &wgpu_device),
            false,
        );
        let result = wgpu_client.apply_rope(&x_w, &cos_w, &sin_w).unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope WGPU vs CPU",
        );
    });
}

#[test]
fn test_apply_rope_interleaved_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (2, 4, 8, 32);

    let x_data = det_tensor(&[b, h, s, d], &cpu_device);
    let cos_data = det_tensor(&[s, d / 2], &cpu_device);
    let sin_data = det_tensor(&[s, d / 2], &cpu_device);

    let x = Var::<CpuRuntime>::new(x_data.clone(), false);
    let cos = Var::<CpuRuntime>::new(cos_data.clone(), false);
    let sin = Var::<CpuRuntime>::new(sin_data.clone(), false);

    let cpu_result = cpu_client.apply_rope_interleaved(&x, &cos, &sin).unwrap();
    let cpu_result_vec = cpu_result.tensor().to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::rope::RoPEOps as _;
        use numr::autograd::Var;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;
        let x_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&x_data.to_vec::<f32>(), &[b, h, s, d], &cuda_device),
            false,
        );
        let cos_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[s, d / 2], &cuda_device),
            false,
        );
        let sin_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[s, d / 2], &cuda_device),
            false,
        );
        let result = cuda_client
            .apply_rope_interleaved(&x_c, &cos_c, &sin_c)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope_interleaved CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::position::rope::RoPEOps as _;
        use numr::autograd::Var;
        use numr::runtime::wgpu::WgpuRuntime;
        use numr::tensor::Tensor;
        let x_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&x_data.to_vec::<f32>(), &[b, h, s, d], &wgpu_device),
            false,
        );
        let cos_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[s, d / 2], &wgpu_device),
            false,
        );
        let sin_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[s, d / 2], &wgpu_device),
            false,
        );
        let result = wgpu_client
            .apply_rope_interleaved(&x_w, &cos_w, &sin_w)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope_interleaved WGPU vs CPU",
        );
    });
}

#[test]
fn test_apply_rope_yarn_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (2, 4, 8, 32);
    let attn_scale = 1.0826f32; // typical YaRN scale for 8x context extension

    let x_data = det_tensor(&[b, h, s, d], &cpu_device);
    let cos_data = det_tensor(&[s, d / 2], &cpu_device);
    let sin_data = det_tensor(&[s, d / 2], &cpu_device);

    let x = Var::<CpuRuntime>::new(x_data.clone(), false);
    let cos = Var::<CpuRuntime>::new(cos_data.clone(), false);
    let sin = Var::<CpuRuntime>::new(sin_data.clone(), false);

    let cpu_result = cpu_client
        .apply_rope_yarn(&x, &cos, &sin, attn_scale)
        .unwrap();
    let cpu_result_vec = cpu_result.tensor().to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::rope::RoPEOps as _;
        use numr::autograd::Var;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;
        let x_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&x_data.to_vec::<f32>(), &[b, h, s, d], &cuda_device),
            false,
        );
        let cos_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[s, d / 2], &cuda_device),
            false,
        );
        let sin_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[s, d / 2], &cuda_device),
            false,
        );
        let result = cuda_client
            .apply_rope_yarn(&x_c, &cos_c, &sin_c, attn_scale)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope_yarn CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::position::rope::RoPEOps as _;
        use numr::autograd::Var;
        use numr::runtime::wgpu::WgpuRuntime;
        use numr::tensor::Tensor;
        let x_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&x_data.to_vec::<f32>(), &[b, h, s, d], &wgpu_device),
            false,
        );
        let cos_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[s, d / 2], &wgpu_device),
            false,
        );
        let sin_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[s, d / 2], &wgpu_device),
            false,
        );
        let result = wgpu_client
            .apply_rope_yarn(&x_w, &cos_w, &sin_w, attn_scale)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope_yarn WGPU vs CPU",
        );
    });
}

#[test]
fn test_apply_rope_yarn_unit_scale_matches_standard() {
    // With attn_scale=1.0 and same caches, YaRN should produce identical output to standard RoPE
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 4, 16);

    let x_data = det_tensor(&[b, h, s, d], &cpu_device);
    let cos_data = det_tensor(&[s, d / 2], &cpu_device);
    let sin_data = det_tensor(&[s, d / 2], &cpu_device);

    let x = Var::<CpuRuntime>::new(x_data.clone(), false);
    let cos = Var::<CpuRuntime>::new(cos_data.clone(), false);
    let sin = Var::<CpuRuntime>::new(sin_data.clone(), false);

    let rope_result = cpu_client.apply_rope(&x, &cos, &sin).unwrap();
    let yarn_result = cpu_client.apply_rope_yarn(&x, &cos, &sin, 1.0).unwrap();

    assert_parity_f32(
        &yarn_result.tensor().to_vec::<f32>(),
        &rope_result.tensor().to_vec::<f32>(),
        "apply_rope_yarn(scale=1.0) vs apply_rope",
    );
}

#[test]
fn test_interleaved_vs_standard_different_outputs() {
    // Interleaved and standard RoPE should produce DIFFERENT outputs for the same input
    // (they pair dimensions differently)
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 1, 2, 8);

    let x_data = det_tensor(&[b, h, s, d], &cpu_device);
    // Use non-trivial cos/sin so rotation actually changes values
    let cos_data: Vec<f32> = (0..s * d / 2).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin_data: Vec<f32> = (0..s * d / 2).map(|i| (i as f32 * 0.3).sin()).collect();

    let cos = Var::<CpuRuntime>::new(
        numr::tensor::Tensor::from_slice(&cos_data, &[s, d / 2], &cpu_device),
        false,
    );
    let sin = Var::<CpuRuntime>::new(
        numr::tensor::Tensor::from_slice(&sin_data, &[s, d / 2], &cpu_device),
        false,
    );
    let x = Var::<CpuRuntime>::new(x_data, false);

    let standard = cpu_client.apply_rope(&x, &cos, &sin).unwrap();
    let interleaved = cpu_client.apply_rope_interleaved(&x, &cos, &sin).unwrap();

    let std_vec = standard.tensor().to_vec::<f32>();
    let int_vec = interleaved.tensor().to_vec::<f32>();

    // They should differ (different pairing strategies)
    let any_different = std_vec
        .iter()
        .zip(int_vec.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_different,
        "standard and interleaved RoPE produced identical outputs — they shouldn't"
    );
}
