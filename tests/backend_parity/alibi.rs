//! Backend parity tests for AlibiOps.

use super::helpers::*;
use boostr::ops::traits::position::alibi::AlibiOps;

#[test]
fn test_alibi_add_bias_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk) = (1, 4, 8, 8);

    // Initialize scores to zero so we can see the bias values
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device);

    cpu_client.alibi_add_bias(&scores, b, h, sq, sk).unwrap();
    let cpu_scores_vec = scores.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::tensor::Tensor;
        let s = Tensor::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        );
        cuda_client.alibi_add_bias(&s, b, h, sq, sk).unwrap();
        assert_parity_f32(
            &s.to_vec::<f32>(),
            &cpu_scores_vec,
            "alibi_add_bias CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::tensor::Tensor;
        let s = Tensor::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &wgpu_device,
        );
        wgpu_client.alibi_add_bias(&s, b, h, sq, sk).unwrap();
        assert_parity_f32(
            &s.to_vec::<f32>(),
            &cpu_scores_vec,
            "alibi_add_bias WGPU vs CPU",
        );
    });
}
