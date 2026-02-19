use boostr::nn::{Mla, MlaConfig};
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

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

#[test]
fn test_mla_forward_no_q_lora() {
    let (client, device) = cpu_setup();
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
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &device).unwrap();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 4 * 16], &[1, 4, 16], &device),
        false,
    );
    let out = mla.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 16]);
}

#[test]
fn test_mla_forward_with_q_lora() {
    let (client, device) = cpu_setup();
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
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &device).unwrap();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 4 * 16], &[1, 4, 16], &device),
        false,
    );
    let out = mla.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 16]);
}

#[test]
fn test_mla_forward_different_head_dim_v() {
    let (client, device) = cpu_setup();
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
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &device).unwrap();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 3 * 16], &[1, 3, 16], &device),
        false,
    );
    let out = mla.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 16]);
}

#[test]
fn test_mla_output_finite() {
    let (client, device) = cpu_setup();
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
    let mla = Mla::<CpuRuntime>::from_config(&cfg, &device).unwrap();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 2 * 8], &[1, 2, 8], &device),
        false,
    );
    let out = mla.forward(&client, &x).unwrap();
    let data: Vec<f32> = out.tensor().to_vec();
    for v in &data {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}
