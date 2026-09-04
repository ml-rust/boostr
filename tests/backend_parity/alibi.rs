//! Backend parity tests for AlibiOps.

use super::helpers::*;
use boostr::ops::traits::position::alibi::AlibiOps;

#[test]
fn test_alibi_add_bias_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk) = (1, 4, 8, 8);

    // Initialize scores to zero so we can see the bias values
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device).unwrap();

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
        )
        .unwrap();
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
        )
        .unwrap();
        wgpu_client.alibi_add_bias(&s, b, h, sq, sk).unwrap();
        assert_parity_f32(
            &s.to_vec::<f32>(),
            &cpu_scores_vec,
            "alibi_add_bias WGPU vs CPU",
        );
    });
}

// `alibi_bf16.cu` is a separate translation unit from `alibi.cu`, resolved
// through its own `ALIBI_BF16_MODULE` constant (src/ops/cuda/kernels/constants.rs).
// A dispatch bug pointing BF16 at the wrong module fails at runtime with a
// missing-symbol lookup, and nothing else in this crate exercises that path.
#[test]
fn test_alibi_add_bias_bf16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk) = (1, 4, 8, 8);

    // Reference computed on CPU in F32, per house rule for BF16 fixtures.
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device).unwrap();
    cpu_client.alibi_add_bias(&scores, b, h, sq, sk).unwrap();
    let cpu_scores_vec = scores.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::dtype::DType;
        use numr::runtime::Device;
        use numr::tensor::Tensor;

        // Gate on the capability, never on the returned error. A device below
        // sm_80 returns KernelError here, but so does a missing-symbol lookup
        // from a mis-wired module — the defect this test exists to catch.
        // Matching on the error would skip on that defect and report it as old
        // hardware.
        if !numr::runtime::cuda::CudaDevice::new(cuda_device.id())
            .profile()
            .caps
            .bf16
        {
            println!(
                "!! test_alibi_add_bias_bf16_cuda SKIPPED: this GPU predates sm_80, which BF16 \
                 ALiBi requires. NOTHING WAS VERIFIED."
            );
            eprintln!(
                "!! test_alibi_add_bias_bf16_cuda SKIPPED: this GPU predates sm_80, which BF16 \
                 ALiBi requires. NOTHING WAS VERIFIED."
            );
            return;
        }

        let s_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        let s_bf16 = s_f32
            .to_dtype(DType::BF16)
            .expect("cast zeros fixture to BF16");

        cuda_client
            .alibi_add_bias(&s_bf16, b, h, sq, sk)
            .expect("BF16 alibi_add_bias must succeed on an sm_80+ device");

        let result_f32 = s_bf16
            .to_dtype(DType::F32)
            .expect("cast BF16 result back to F32 for comparison")
            .to_vec::<f32>();
        // BF16 keeps ~8 mantissa bits, so tolerance is set by the dtype, not
        // by the op.
        assert_parity_f32_tol(
            &result_f32,
            &cpu_scores_vec,
            "alibi_add_bias BF16 CUDA vs CPU",
            4e-2,
            2e-2,
        );
    });
}

#[test]
fn test_alibi_add_bias_causal_bf16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk, position) = (1, 4, 8, 16, 8);

    // Reference computed on CPU in F32, per house rule for BF16 fixtures.
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device).unwrap();
    cpu_client
        .alibi_add_bias_causal(&scores, b, h, sq, sk, position)
        .unwrap();
    let cpu_scores_vec = scores.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::dtype::DType;
        use numr::runtime::Device;
        use numr::tensor::Tensor;

        // Same reasoning as the non-causal test above: gate on the capability,
        // never on the returned error.
        if !numr::runtime::cuda::CudaDevice::new(cuda_device.id())
            .profile()
            .caps
            .bf16
        {
            println!(
                "!! test_alibi_add_bias_causal_bf16_cuda SKIPPED: this GPU predates sm_80, which \
                 BF16 ALiBi requires. NOTHING WAS VERIFIED."
            );
            eprintln!(
                "!! test_alibi_add_bias_causal_bf16_cuda SKIPPED: this GPU predates sm_80, which \
                 BF16 ALiBi requires. NOTHING WAS VERIFIED."
            );
            return;
        }

        let s_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        let s_bf16 = s_f32
            .to_dtype(DType::BF16)
            .expect("cast zeros fixture to BF16");

        cuda_client
            .alibi_add_bias_causal(&s_bf16, b, h, sq, sk, position)
            .expect("BF16 alibi_add_bias_causal must succeed on an sm_80+ device");

        let result_f32 = s_bf16
            .to_dtype(DType::F32)
            .expect("cast BF16 result back to F32 for comparison")
            .to_vec::<f32>();
        // BF16 keeps ~8 mantissa bits, so tolerance is set by the dtype, not
        // by the op.
        assert_parity_f32_tol(
            &result_f32,
            &cpu_scores_vec,
            "alibi_add_bias_causal BF16 CUDA vs CPU",
            4e-2,
            2e-2,
        );
    });
}
