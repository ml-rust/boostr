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
