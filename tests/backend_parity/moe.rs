//! Backend parity tests for MoEOps.

use super::helpers::*;
use boostr::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[test]
fn test_moe_top_k_routing_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 8;
    let num_experts = 4;
    let k = 2;

    let logits_data = det_tensor(&[num_tokens, num_experts], &cpu_device);
    let (_cpu_indices, cpu_weights) = cpu_client.moe_top_k_routing(&logits_data, k).unwrap();

    let cpu_weights_vec = cpu_weights.to_vec::<f32>();

    // Verify weights sum to 1 per token
    for t in 0..num_tokens {
        let sum: f32 = (0..k).map(|j| cpu_weights_vec[t * k + j]).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "CPU token {} weights sum={}, expected 1.0",
            t,
            sum
        );
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let logits_c = Tensor::from_slice(
            &logits_data.to_vec::<f32>(),
            &[num_tokens, num_experts],
            &cuda_device,
        );
        let (_indices_c, weights_c) = cuda_client.moe_top_k_routing(&logits_c, k).unwrap();
        assert_parity_f32(
            &weights_c.to_vec::<f32>(),
            &cpu_weights_vec,
            "moe_top_k_routing weights CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let logits_w = Tensor::from_slice(
            &logits_data.to_vec::<f32>(),
            &[num_tokens, num_experts],
            &wgpu_device,
        );
        let (_indices_w, weights_w) = wgpu_client.moe_top_k_routing(&logits_w, k).unwrap();
        assert_parity_f32(
            &weights_w.to_vec::<f32>(),
            &cpu_weights_vec,
            "moe_top_k_routing weights WGPU vs CPU",
        );
    });
}

#[test]
fn test_moe_permute_tokens_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 6;
    let hidden_dim = 16;
    let num_experts = 4;
    let k = 2;

    let tokens_data = det_tensor(&[num_tokens, hidden_dim], &cpu_device);
    // Use I32 indices for WebGPU compatibility
    let indices_data: Vec<i32> = vec![0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3];
    let indices = Tensor::<CpuRuntime>::from_slice(&indices_data, &[num_tokens, k], &cpu_device);

    let (cpu_permuted, _cpu_offsets, _cpu_sort) = cpu_client
        .moe_permute_tokens(&tokens_data, &indices, num_experts)
        .unwrap();

    let cpu_permuted_vec = cpu_permuted.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_c = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[num_tokens, hidden_dim],
            &cuda_device,
        );
        let indices_c = Tensor::from_slice(&indices_data, &[num_tokens, k], &cuda_device);
        let (permuted_c, _, _) = cuda_client
            .moe_permute_tokens(&tokens_c, &indices_c, num_experts)
            .unwrap();
        assert_parity_f32(
            &permuted_c.to_vec::<f32>(),
            &cpu_permuted_vec,
            "moe_permute_tokens CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_w = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[num_tokens, hidden_dim],
            &wgpu_device,
        );
        let indices_w = Tensor::from_slice(&indices_data, &[num_tokens, k], &wgpu_device);
        let (permuted_w, _, _) = wgpu_client
            .moe_permute_tokens(&tokens_w, &indices_w, num_experts)
            .unwrap();
        assert_parity_f32(
            &permuted_w.to_vec::<f32>(),
            &cpu_permuted_vec,
            "moe_permute_tokens WGPU vs CPU",
        );
    });
}

#[test]
fn test_moe_unpermute_tokens_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 4;
    let hidden_dim = 8;
    let num_experts = 3;
    let k = 2;

    let tokens_data = det_tensor(&[num_tokens, hidden_dim], &cpu_device);
    let indices_data: Vec<i32> = vec![0, 1, 2, 0, 1, 2, 0, 1];
    let indices = Tensor::<CpuRuntime>::from_slice(&indices_data, &[num_tokens, k], &cpu_device);
    let weights_data: Vec<f32> = vec![0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.4, 0.6];
    let weights = Tensor::<CpuRuntime>::from_slice(&weights_data, &[num_tokens, k], &cpu_device);

    let (permuted, _, sort_indices) = cpu_client
        .moe_permute_tokens(&tokens_data, &indices, num_experts)
        .unwrap();

    // Use permuted as expert_output (identity transform)
    let cpu_result = cpu_client
        .moe_unpermute_tokens(&permuted, &sort_indices, &weights, num_tokens)
        .unwrap();
    let cpu_result_vec = cpu_result.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_c = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[num_tokens, hidden_dim],
            &cuda_device,
        );
        let indices_c = Tensor::from_slice(&indices_data, &[num_tokens, k], &cuda_device);
        let weights_c = Tensor::from_slice(&weights_data, &[num_tokens, k], &cuda_device);
        let (permuted_c, _, sort_c) = cuda_client
            .moe_permute_tokens(&tokens_c, &indices_c, num_experts)
            .unwrap();
        let result_c = cuda_client
            .moe_unpermute_tokens(&permuted_c, &sort_c, &weights_c, num_tokens)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_c.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_unpermute_tokens CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_w = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[num_tokens, hidden_dim],
            &wgpu_device,
        );
        let indices_w = Tensor::from_slice(&indices_data, &[num_tokens, k], &wgpu_device);
        let weights_w = Tensor::from_slice(&weights_data, &[num_tokens, k], &wgpu_device);
        let (permuted_w, _, sort_w) = wgpu_client
            .moe_permute_tokens(&tokens_w, &indices_w, num_experts)
            .unwrap();
        let result_w = wgpu_client
            .moe_unpermute_tokens(&permuted_w, &sort_w, &weights_w, num_tokens)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_w.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_unpermute_tokens WGPU vs CPU",
        );
    });
}

#[test]
fn test_moe_grouped_gemm_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_experts = 3;
    let in_dim = 8;
    let out_dim = 4;
    let total_tokens = 9;

    let tokens_data = det_tensor(&[total_tokens, in_dim], &cpu_device);
    let weights_data = det_tensor(&[num_experts, in_dim, out_dim], &cpu_device);
    let offsets_data: Vec<i32> = vec![0, 3, 6, 9];
    let offsets = Tensor::<CpuRuntime>::from_slice(&offsets_data, &[num_experts + 1], &cpu_device);

    let cpu_result = cpu_client
        .moe_grouped_gemm(&tokens_data, &weights_data, &offsets)
        .unwrap();
    let cpu_result_vec = cpu_result.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_c = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[total_tokens, in_dim],
            &cuda_device,
        );
        let weights_c = Tensor::from_slice(
            &weights_data.to_vec::<f32>(),
            &[num_experts, in_dim, out_dim],
            &cuda_device,
        );
        let offsets_c = Tensor::from_slice(&offsets_data, &[num_experts + 1], &cuda_device);
        let result_c = cuda_client
            .moe_grouped_gemm(&tokens_c, &weights_c, &offsets_c)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_c.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_grouped_gemm CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_w = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[total_tokens, in_dim],
            &wgpu_device,
        );
        let weights_w = Tensor::from_slice(
            &weights_data.to_vec::<f32>(),
            &[num_experts, in_dim, out_dim],
            &wgpu_device,
        );
        let offsets_w = Tensor::from_slice(&offsets_data, &[num_experts + 1], &wgpu_device);
        let result_w = wgpu_client
            .moe_grouped_gemm(&tokens_w, &weights_w, &offsets_w)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_w.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_grouped_gemm WGPU vs CPU",
        );
    });
}

#[test]
fn test_moe_grouped_gemm_fused_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_experts = 2;
    let in_dim = 8;
    let out_dim = 4;
    let total_tokens = 6;

    let tokens_data = det_tensor(&[total_tokens, in_dim], &cpu_device);
    let weights_data = det_tensor(&[num_experts, in_dim, out_dim], &cpu_device);
    let offsets_data: Vec<i32> = vec![0, 3, 6];
    let offsets = Tensor::<CpuRuntime>::from_slice(&offsets_data, &[num_experts + 1], &cpu_device);

    let cpu_result = cpu_client
        .moe_grouped_gemm_fused(&tokens_data, &weights_data, &offsets, MoEActivation::SiLU)
        .unwrap();
    let cpu_result_vec = cpu_result.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_c = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[total_tokens, in_dim],
            &cuda_device,
        );
        let weights_c = Tensor::from_slice(
            &weights_data.to_vec::<f32>(),
            &[num_experts, in_dim, out_dim],
            &cuda_device,
        );
        let offsets_c = Tensor::from_slice(&offsets_data, &[num_experts + 1], &cuda_device);
        let result_c = cuda_client
            .moe_grouped_gemm_fused(&tokens_c, &weights_c, &offsets_c, MoEActivation::SiLU)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_c.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_grouped_gemm_fused SiLU CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_w = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[total_tokens, in_dim],
            &wgpu_device,
        );
        let weights_w = Tensor::from_slice(
            &weights_data.to_vec::<f32>(),
            &[num_experts, in_dim, out_dim],
            &wgpu_device,
        );
        let offsets_w = Tensor::from_slice(&offsets_data, &[num_experts + 1], &wgpu_device);
        let result_w = wgpu_client
            .moe_grouped_gemm_fused(&tokens_w, &weights_w, &offsets_w, MoEActivation::SiLU)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_w.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_grouped_gemm_fused SiLU WGPU vs CPU",
        );
    });
}

#[test]
fn test_moe_grouped_gemm_fused_gelu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_experts = 2;
    let in_dim = 8;
    let out_dim = 4;
    let total_tokens = 6;

    let tokens_data = det_tensor(&[total_tokens, in_dim], &cpu_device);
    let weights_data = det_tensor(&[num_experts, in_dim, out_dim], &cpu_device);
    let offsets_data: Vec<i32> = vec![0, 3, 6];
    let offsets = Tensor::<CpuRuntime>::from_slice(&offsets_data, &[num_experts + 1], &cpu_device);

    let cpu_result = cpu_client
        .moe_grouped_gemm_fused(&tokens_data, &weights_data, &offsets, MoEActivation::GeLU)
        .unwrap();
    let cpu_result_vec = cpu_result.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_c = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[total_tokens, in_dim],
            &cuda_device,
        );
        let weights_c = Tensor::from_slice(
            &weights_data.to_vec::<f32>(),
            &[num_experts, in_dim, out_dim],
            &cuda_device,
        );
        let offsets_c = Tensor::from_slice(&offsets_data, &[num_experts + 1], &cuda_device);
        let result_c = cuda_client
            .moe_grouped_gemm_fused(&tokens_c, &weights_c, &offsets_c, MoEActivation::GeLU)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_c.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_grouped_gemm_fused GeLU CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let tokens_w = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[total_tokens, in_dim],
            &wgpu_device,
        );
        let weights_w = Tensor::from_slice(
            &weights_data.to_vec::<f32>(),
            &[num_experts, in_dim, out_dim],
            &wgpu_device,
        );
        let offsets_w = Tensor::from_slice(&offsets_data, &[num_experts + 1], &wgpu_device);
        let result_w = wgpu_client
            .moe_grouped_gemm_fused(&tokens_w, &weights_w, &offsets_w, MoEActivation::GeLU)
            .unwrap();
        assert_parity_f32_relaxed(
            &result_w.to_vec::<f32>(),
            &cpu_result_vec,
            "moe_grouped_gemm_fused GeLU WGPU vs CPU",
        );
    });
}

#[test]
fn test_moe_end_to_end_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let num_tokens = 6;
    let hidden_dim = 16;
    let num_experts = 4;
    let k = 2;
    let out_dim = 8;

    // Step 1: routing
    let logits_data = det_tensor(&[num_tokens, num_experts], &cpu_device);
    let (cpu_indices, cpu_weights) = cpu_client.moe_top_k_routing(&logits_data, k).unwrap();

    // Step 2: permute
    let tokens_data = det_tensor(&[num_tokens, hidden_dim], &cpu_device);
    let (cpu_permuted, cpu_offsets, cpu_sort) = cpu_client
        .moe_permute_tokens(&tokens_data, &cpu_indices, num_experts)
        .unwrap();

    // Step 3: grouped gemm
    let expert_weights_data = det_tensor(&[num_experts, hidden_dim, out_dim], &cpu_device);
    let cpu_expert_out = cpu_client
        .moe_grouped_gemm(&cpu_permuted, &expert_weights_data, &cpu_offsets)
        .unwrap();

    // Step 4: unpermute
    let cpu_final = cpu_client
        .moe_unpermute_tokens(&cpu_expert_out, &cpu_sort, &cpu_weights, num_tokens)
        .unwrap();
    let cpu_final_vec = cpu_final.to_vec::<f32>();

    assert_eq!(cpu_final.shape(), &[num_tokens, out_dim]);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let logits_c = Tensor::from_slice(
            &logits_data.to_vec::<f32>(),
            &[num_tokens, num_experts],
            &cuda_device,
        );
        let tokens_c = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[num_tokens, hidden_dim],
            &cuda_device,
        );
        let ew_c = Tensor::from_slice(
            &expert_weights_data.to_vec::<f32>(),
            &[num_experts, hidden_dim, out_dim],
            &cuda_device,
        );

        let (idx_c, wt_c) = cuda_client.moe_top_k_routing(&logits_c, k).unwrap();
        let (perm_c, off_c, sort_c) = cuda_client
            .moe_permute_tokens(&tokens_c, &idx_c, num_experts)
            .unwrap();
        let expert_out_c = cuda_client
            .moe_grouped_gemm(&perm_c, &ew_c, &off_c)
            .unwrap();
        let final_c = cuda_client
            .moe_unpermute_tokens(&expert_out_c, &sort_c, &wt_c, num_tokens)
            .unwrap();

        assert_parity_f32_relaxed(
            &final_c.to_vec::<f32>(),
            &cpu_final_vec,
            "moe end-to-end CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::moe::MoEOps as _;
        let logits_w = Tensor::from_slice(
            &logits_data.to_vec::<f32>(),
            &[num_tokens, num_experts],
            &wgpu_device,
        );
        let tokens_w = Tensor::from_slice(
            &tokens_data.to_vec::<f32>(),
            &[num_tokens, hidden_dim],
            &wgpu_device,
        );
        let ew_w = Tensor::from_slice(
            &expert_weights_data.to_vec::<f32>(),
            &[num_experts, hidden_dim, out_dim],
            &wgpu_device,
        );

        let (idx_w, wt_w) = wgpu_client.moe_top_k_routing(&logits_w, k).unwrap();
        let (perm_w, off_w, sort_w) = wgpu_client
            .moe_permute_tokens(&tokens_w, &idx_w, num_experts)
            .unwrap();
        let expert_out_w = wgpu_client
            .moe_grouped_gemm(&perm_w, &ew_w, &off_w)
            .unwrap();
        let final_w = wgpu_client
            .moe_unpermute_tokens(&expert_out_w, &sort_w, &wt_w, num_tokens)
            .unwrap();

        assert_parity_f32_relaxed(
            &final_w.to_vec::<f32>(),
            &cpu_final_vec,
            "moe end-to-end WGPU vs CPU",
        );
    });
}
