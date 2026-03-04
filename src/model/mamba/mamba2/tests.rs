use crate::model::mamba::mamba2::config::Mamba2Config;
use crate::model::mamba::mamba2::layer::{Mamba2, Mamba2Weights};
use crate::nn::{Conv1d, Linear};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::ops::PaddingMode;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

fn tiny_mamba2() -> (Mamba2<CpuRuntime>, Mamba2Config) {
    let (_, device) = cpu_setup();
    let config = Mamba2Config::new(8)
        .with_nheads(1)
        .with_d_state(4)
        .with_expand(2)
        .with_dt_softplus(false)
        .with_use_dt_bias(false)
        .with_use_d(false);

    let d_inner = config.d_inner();
    let conv_channels = config.conv_channels();
    let proj_dim = config.proj_dim();

    let in_proj = Linear::new(
        Tensor::<CpuRuntime>::from_slice(&[0.01f32; 328], &[proj_dim, 8], &device),
        None,
        false,
    );
    let conv1d = Conv1d::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device),
        None,
        1,
        PaddingMode::Custom(3, 0, 0, 0),
        1,
        conv_channels,
        false,
    );
    let out_proj = Linear::new(
        Tensor::<CpuRuntime>::from_slice(&[0.01f32; 128], &[8, d_inner], &device),
        None,
        false,
    );
    let a_log = Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[config.nheads], &device);

    let weights = Mamba2Weights {
        in_proj,
        conv1d,
        out_proj,
        a_log,
        dt_bias: None,
        d_param: None,
        norm: None,
    };
    let mamba = Mamba2::new(config.clone(), weights, false);
    (mamba, config)
}

#[test]
fn test_mamba2_forward_shape() {
    let (client, device) = cpu_setup();
    let (mamba, _) = tiny_mamba2();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[1, 4, 8], &device),
        false,
    );

    let out = mamba.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 8]);
}

#[test]
fn test_mamba2_forward_invalid_input() {
    let (client, device) = cpu_setup();
    let (mamba, _) = tiny_mamba2();

    // 2D input should fail
    let x_2d = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[1, 8], &device),
        false,
    );
    assert!(mamba.forward(&client, &x_2d).is_err());

    // Wrong d_model should fail
    let x_wrong = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 4, 3], &device),
        false,
    );
    assert!(mamba.forward(&client, &x_wrong).is_err());
}
