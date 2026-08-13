use crate::model::mamba::mamba3::config::Mamba3Config;
use crate::model::mamba::mamba3::layer::{Mamba3, Mamba3Weights};
use crate::nn::{Linear, RmsNorm};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

fn linear_with_value(rows: usize, cols: usize, value: f32) -> Linear<CpuRuntime> {
    let (_, device) = cpu_setup();
    Linear::new(
        Tensor::<CpuRuntime>::from_slice(&vec![value; rows * cols], &[rows, cols], &device),
        None,
        false,
    )
}

fn tiny_mamba3(config: Mamba3Config) -> Mamba3<CpuRuntime> {
    let (_, device) = cpu_setup();
    let in_proj = linear_with_value(config.proj_dim(), config.d_model, 0.0);
    let out_proj = linear_with_value(config.d_model, config.d_inner(), 0.0);
    let lambda_proj = linear_with_value(config.nheads, config.d_model, 0.0);
    let theta_proj = if config.use_complex_rope {
        Some(linear_with_value(
            config.nheads * (config.d_state / 2),
            config.d_model,
            0.0,
        ))
    } else {
        None
    };
    let mimo_x_up = if config.mimo_rank > 0 {
        Some(linear_with_value(
            config.headdim * config.mimo_rank,
            config.headdim,
            0.0,
        ))
    } else {
        None
    };
    let mimo_x_down = if config.mimo_rank > 0 {
        Some(linear_with_value(
            config.headdim,
            config.headdim * config.mimo_rank,
            0.0,
        ))
    } else {
        None
    };

    let weights = Mamba3Weights {
        in_proj,
        out_proj,
        lambda_proj,
        theta_proj,
        b_bias: Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; config.nheads * config.d_state],
            &[config.nheads, config.d_state],
            &device,
        ),
        c_bias: Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; config.nheads * config.d_state],
            &[config.nheads, config.d_state],
            &device,
        ),
        dt_bias: if config.use_dt_bias {
            Some(Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; config.nheads],
                &[config.nheads],
                &device,
            ))
        } else {
            None
        },
        a_log: Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; config.nheads],
            &[config.nheads],
            &device,
        ),
        d_param: if config.use_d {
            Some(Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; config.nheads],
                &[config.nheads],
                &device,
            ))
        } else {
            None
        },
        bc_norm: RmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![1.0f32; config.d_state],
                &[config.d_state],
                &device,
            ),
            1e-6,
            false,
        ),
        norm: RmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![1.0f32; config.d_inner()],
                &[config.d_inner()],
                &device,
            ),
            1e-6,
            false,
        ),
        conv1d: None,
        mimo_x_up,
        mimo_x_down,
    };
    Mamba3::new(config, weights, false)
}

#[test]
fn test_mamba3_forward_shape() {
    let (client, device) = cpu_setup();
    let config = Mamba3Config::new(4)
        .with_nheads(1)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_dt_bias(false)
        .with_use_d(false);
    let mamba = tiny_mamba3(config);
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 3, 4], &device),
        false,
    );

    let out = mamba.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 4]);
}

#[test]
fn test_mamba3_forward_invalid_input() {
    let (client, device) = cpu_setup();
    let config = Mamba3Config::new(4)
        .with_nheads(1)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_dt_bias(false)
        .with_use_d(false);
    let mamba = tiny_mamba3(config);

    let x_2d = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 4], &[1, 4], &device),
        false,
    );
    assert!(mamba.forward(&client, &x_2d).is_err());

    let x_wrong = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[1, 3, 2], &device),
        false,
    );
    assert!(mamba.forward(&client, &x_wrong).is_err());
}

#[test]
fn test_trapezoidal_discretization_matches_f64_reference() {
    let (client, device) = cpu_setup();
    let config = Mamba3Config::new(2)
        .with_nheads(1)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_dt_bias(false)
        .with_use_d(true);
    let mut mamba = tiny_mamba3(config);
    mamba.a_log = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.5f32.ln()], &[1], &device),
        false,
    );
    mamba.d_param = Some(Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32], &[1], &device),
        false,
    ));

    let x_data = [1.0f32, 0.5, -0.25, 2.0, 0.75, -1.0];
    let b_data = [0.5f32, -0.2, 0.1, 0.3, -0.4, 0.25];
    let c_data = [1.0f32, 0.5, -0.25, 0.75, 0.6, -0.1];
    let dt_data = [0.2f32, 0.4, 0.3];
    let lambda_data = [0.7f32, 0.2, 0.9];

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 3, 1, 2], &device),
        false,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 3, 1, 2], &device),
        false,
    );
    let c = Var::new(
        Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 3, 1, 2], &device),
        false,
    );
    let dt = Var::new(
        Tensor::<CpuRuntime>::from_slice(&dt_data, &[1, 3, 1], &device),
        false,
    );
    let lambda = Var::new(
        Tensor::<CpuRuntime>::from_slice(&lambda_data, &[1, 3, 1], &device),
        false,
    );

    let out = mamba
        .trapezoidal_ssm_forward(&client, &x, &b, &c, &dt, &lambda)
        .unwrap();
    let data: Vec<f32> = out.tensor().to_vec();

    let expected =
        trapezoidal_reference(&x_data, &b_data, &c_data, &dt_data, &lambda_data, -0.5, 0.1);
    for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual as f64 - expected).abs() < 2e-5,
            "idx={i}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn test_complex_rope_matches_f64_reference() {
    let (client, device) = cpu_setup();
    let config = Mamba3Config::new(2)
        .with_nheads(1)
        .with_expand(1)
        .with_d_state(4)
        .with_complex_rope(true)
        .with_dt_softplus(false)
        .with_use_dt_bias(false)
        .with_use_d(false);
    let mamba = tiny_mamba3(config);

    let tensor_data = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0];
    let angles_data = [
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        std::f32::consts::FRAC_PI_4,
        -std::f32::consts::FRAC_PI_2,
    ];
    let tensor = Var::new(
        Tensor::<CpuRuntime>::from_slice(&tensor_data, &[1, 2, 1, 4], &device),
        false,
    );
    let angles = Var::new(
        Tensor::<CpuRuntime>::from_slice(&angles_data, &[1, 2, 1, 2], &device),
        false,
    );

    let out = mamba.apply_rope(&client, &tensor, &angles).unwrap();
    let data: Vec<f32> = out.tensor().to_vec();
    let expected = rope_reference(&tensor_data, &angles_data);

    for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual as f64 - expected).abs() < 2e-5,
            "idx={i}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn test_mimo_up_down_matches_f64_reference() {
    let (client, device) = cpu_setup();
    let config = Mamba3Config::new(2)
        .with_nheads(1)
        .with_expand(1)
        .with_d_state(2)
        .with_mimo_rank(2)
        .with_dt_softplus(false)
        .with_use_dt_bias(false)
        .with_use_d(false);
    let mut mamba = tiny_mamba3(config);
    mamba.mimo_x_up = Some(Linear::new(
        Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5],
            &[4, 2],
            &device,
        ),
        None,
        false,
    ));
    mamba.mimo_x_down = Some(Linear::new(
        Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0],
            &[2, 4],
            &device,
        ),
        None,
        false,
    ));

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2], &device),
        false,
    );
    let up = mamba.apply_mimo_up(&client, &x, 1, 1).unwrap();
    let up_data: Vec<f32> = up.tensor().to_vec();
    let up_expected = [1.0f64, 2.0, 3.0, 0.0];
    for (i, (actual, expected)) in up_data.iter().zip(up_expected.iter()).enumerate() {
        assert!(
            (*actual as f64 - expected).abs() < 1e-6,
            "up idx={i}: actual={actual}, expected={expected}"
        );
    }

    let down = mamba.apply_mimo_down(&client, &up, 1, 1).unwrap();
    let down_data: Vec<f32> = down.tensor().to_vec();
    let down_expected = [1.0f64, -1.0];
    for (i, (actual, expected)) in down_data.iter().zip(down_expected.iter()).enumerate() {
        assert!(
            (*actual as f64 - expected).abs() < 1e-6,
            "down idx={i}: actual={actual}, expected={expected}"
        );
    }
}

fn trapezoidal_reference(
    x: &[f32; 6],
    b: &[f32; 6],
    c: &[f32; 6],
    dt: &[f32; 3],
    lambda: &[f32; 3],
    a: f64,
    d: f64,
) -> Vec<f64> {
    let mut h = [[0.0f64; 2]; 2];
    let mut prev_x = [0.0f64; 2];
    let mut prev_b = [0.0f64; 2];
    let mut out = Vec::with_capacity(6);

    for t in 0..3 {
        let x_t = [x[2 * t] as f64, x[2 * t + 1] as f64];
        let b_t = [b[2 * t] as f64, b[2 * t + 1] as f64];
        let c_t = [c[2 * t] as f64, c[2 * t + 1] as f64];
        let dt_t = dt[t] as f64;
        let lambda_t = lambda[t] as f64;
        let alpha = (dt_t * a).exp();
        let beta = (1.0 - lambda_t) * dt_t * alpha;
        let gamma = lambda_t * dt_t;

        for dim in 0..2 {
            for state in 0..2 {
                h[dim][state] = alpha * h[dim][state]
                    + beta * prev_x[dim] * prev_b[state]
                    + gamma * x_t[dim] * b_t[state];
            }
        }

        for dim in 0..2 {
            out.push(h[dim][0] * c_t[0] + h[dim][1] * c_t[1] + d * x_t[dim]);
        }
        prev_x = x_t;
        prev_b = b_t;
    }

    out
}

fn rope_reference(tensor: &[f32; 8], angles: &[f32; 4]) -> Vec<f64> {
    let mut out = Vec::with_capacity(8);
    for t in 0..2 {
        for pair in 0..2 {
            let base = t * 4 + pair * 2;
            let real = tensor[base] as f64;
            let imag = tensor[base + 1] as f64;
            let angle = angles[t * 2 + pair] as f64;
            out.push(real * angle.cos() - imag * angle.sin());
            out.push(real * angle.sin() + imag * angle.cos());
        }
    }
    out
}
