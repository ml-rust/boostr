//! Backend parity tests for MlaOps and Mla nn module.

use super::helpers::*;
use boostr::nn::{Mla, MlaConfig};
use boostr::ops::traits::attention::mla::MlaOps;
use numr::autograd::Var;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn test_mla_scaled_dot_product_attention_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d_k, d_v) = (1, 2, 8, 20, 16);
    let scale = (d_k as f64).sqrt().recip();

    let q_data = det_tensor(&[b, h, s, d_k], &cpu_device);
    let k_data = det_tensor(&[b, h, s, d_k], &cpu_device);
    let v_data = det_tensor(&[b, h, s, d_v], &cpu_device);

    let q = Var::<CpuRuntime>::new(q_data.clone(), false);
    let k = Var::<CpuRuntime>::new(k_data.clone(), false);
    let v = Var::<CpuRuntime>::new(v_data.clone(), false);

    let cpu_result = cpu_client
        .scaled_dot_product_attention(&q, &k, &v, scale, false)
        .unwrap();
    let cpu_result_vec = cpu_result.tensor().to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::mla::MlaOps as _;
        use numr::autograd::Var;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;
        let q_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&q_data.to_vec::<f32>(), &[b, h, s, d_k], &cuda_device),
            false,
        );
        let k_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&k_data.to_vec::<f32>(), &[b, h, s, d_k], &cuda_device),
            false,
        );
        let v_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&v_data.to_vec::<f32>(), &[b, h, s, d_v], &cuda_device),
            false,
        );
        let result = cuda_client
            .scaled_dot_product_attention(&q_c, &k_c, &v_c, scale, false)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "mla_sdpa CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::mla::MlaOps as _;
        use numr::autograd::Var;
        use numr::runtime::wgpu::WgpuRuntime;
        use numr::tensor::Tensor;
        let q_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&q_data.to_vec::<f32>(), &[b, h, s, d_k], &wgpu_device),
            false,
        );
        let k_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&k_data.to_vec::<f32>(), &[b, h, s, d_k], &wgpu_device),
            false,
        );
        let v_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&v_data.to_vec::<f32>(), &[b, h, s, d_v], &wgpu_device),
            false,
        );
        let result = wgpu_client
            .scaled_dot_product_attention(&q_w, &k_w, &v_w, scale, false)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "mla_sdpa WGPU vs CPU",
        );
    });
}

#[test]
fn test_mla_causal_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d_k, d_v) = (1, 2, 8, 20, 16);
    let scale = (d_k as f64).sqrt().recip();

    let q_data = det_tensor(&[b, h, s, d_k], &cpu_device);
    let k_data = det_tensor(&[b, h, s, d_k], &cpu_device);
    let v_data = det_tensor(&[b, h, s, d_v], &cpu_device);

    let q = Var::<CpuRuntime>::new(q_data.clone(), false);
    let k = Var::<CpuRuntime>::new(k_data.clone(), false);
    let v = Var::<CpuRuntime>::new(v_data.clone(), false);

    let cpu_result = cpu_client
        .scaled_dot_product_attention(&q, &k, &v, scale, true)
        .unwrap();
    let cpu_result_vec = cpu_result.tensor().to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::mla::MlaOps as _;
        use numr::autograd::Var;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;
        let q_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&q_data.to_vec::<f32>(), &[b, h, s, d_k], &cuda_device),
            false,
        );
        let k_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&k_data.to_vec::<f32>(), &[b, h, s, d_k], &cuda_device),
            false,
        );
        let v_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&v_data.to_vec::<f32>(), &[b, h, s, d_v], &cuda_device),
            false,
        );
        let result = cuda_client
            .scaled_dot_product_attention(&q_c, &k_c, &v_c, scale, true)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "mla_sdpa causal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::mla::MlaOps as _;
        use numr::autograd::Var;
        use numr::runtime::wgpu::WgpuRuntime;
        use numr::tensor::Tensor;
        let q_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&q_data.to_vec::<f32>(), &[b, h, s, d_k], &wgpu_device),
            false,
        );
        let k_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&k_data.to_vec::<f32>(), &[b, h, s, d_k], &wgpu_device),
            false,
        );
        let v_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&v_data.to_vec::<f32>(), &[b, h, s, d_v], &wgpu_device),
            false,
        );
        let result = wgpu_client
            .scaled_dot_product_attention(&q_w, &k_w, &v_w, scale, true)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "mla_sdpa causal WGPU vs CPU",
        );
    });
}

// ---------------------------------------------------------------------------
// MlaConfig tests — pure config validation, no tensor ops.
// ---------------------------------------------------------------------------

#[test]
fn test_mla_config_defaults() {
    let cfg = MlaConfig::deepseek_v2(4096, 32, 512, 1536, 64, 8192);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.qk_head_dim(), 192);
    assert!(cfg.q_uses_lora());
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_mla_config_validation() {
    let cfg = MlaConfig {
        kv_lora_rank: 0,
        ..MlaConfig::deepseek_v2(64, 4, 16, 0, 8, 32)
    };
    assert!(cfg.validate().is_err());

    let cfg = MlaConfig {
        rope_head_dim: 256,
        head_dim: 16,
        ..MlaConfig::deepseek_v2(64, 4, 16, 0, 8, 32)
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Mla nn module forward tests — CPU-only shape and correctness checks.
//
// These tests verify the Mla module's forward pass produces correct output
// shapes and finite values. Cross-backend weight-level parity is not tested
// here because Mla::from_config initialises random weights independently on
// each backend, so outputs will differ; op-level parity is covered above via
// scaled_dot_product_attention parity tests.
// ---------------------------------------------------------------------------

#[test]
fn test_mla_forward_no_q_lora() {
    use numr::tensor::Tensor;
    let (cpu_client, cpu_device) = setup_cpu();
    let cfg = MlaConfig {
        hidden_size: 16,
        num_heads: 2,
        head_dim: 8,
        head_dim_v: 8,
        kv_lora_rank: 8,
        q_lora_rank: 0,
        rope_head_dim: 4,
        max_seq_len: 32,
        rope_theta: 10000.0,
        use_norm: true,
        norm_eps: 1e-6,
    };
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &cpu_device).unwrap();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 4 * 16], &[1, 4, 16], &cpu_device),
        false,
    );
    let out = mla.forward(&cpu_client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 16]);
}

#[test]
fn test_mla_forward_with_q_lora() {
    use numr::tensor::Tensor;
    let (cpu_client, cpu_device) = setup_cpu();
    let cfg = MlaConfig {
        hidden_size: 16,
        num_heads: 2,
        head_dim: 8,
        head_dim_v: 8,
        kv_lora_rank: 8,
        q_lora_rank: 12,
        rope_head_dim: 4,
        max_seq_len: 32,
        rope_theta: 10000.0,
        use_norm: true,
        norm_eps: 1e-6,
    };
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &cpu_device).unwrap();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 4 * 16], &[1, 4, 16], &cpu_device),
        false,
    );
    let out = mla.forward(&cpu_client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 16]);
}

#[test]
fn test_mla_forward_different_head_dim_v() {
    use numr::tensor::Tensor;
    let (cpu_client, cpu_device) = setup_cpu();
    let cfg = MlaConfig {
        hidden_size: 16,
        num_heads: 2,
        head_dim: 6,
        head_dim_v: 4,
        kv_lora_rank: 8,
        q_lora_rank: 0,
        rope_head_dim: 4,
        max_seq_len: 32,
        rope_theta: 10000.0,
        use_norm: false,
        norm_eps: 1e-6,
    };
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &cpu_device).unwrap();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 3 * 16], &[1, 3, 16], &cpu_device),
        false,
    );
    let out = mla.forward(&cpu_client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 16]);
}

#[test]
fn test_mla_output_finite() {
    use numr::tensor::Tensor;
    let (cpu_client, cpu_device) = setup_cpu();
    let cfg = MlaConfig {
        hidden_size: 8,
        num_heads: 1,
        head_dim: 4,
        head_dim_v: 4,
        kv_lora_rank: 4,
        q_lora_rank: 0,
        rope_head_dim: 4,
        max_seq_len: 16,
        rope_theta: 10000.0,
        use_norm: true,
        norm_eps: 1e-6,
    };
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &cpu_device).unwrap();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 2 * 8], &[1, 2, 8], &cpu_device),
        false,
    );
    let out = mla.forward(&cpu_client, &x).unwrap();
    let data: Vec<f32> = out.tensor().to_vec();
    for v in &data {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}
