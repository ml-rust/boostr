use crate::model::mamba::mamba1::config::Mamba1Config;
use crate::model::mamba::mamba1::layer::{Mamba1, Mamba1Weights};
use crate::nn::{Conv1d, Linear, Module, VarBuilder, VarMap};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::{Runtime, cpu::CpuRuntime};
use numr::tensor::Tensor;

fn linear_with_value(rows: usize, cols: usize, value: f32) -> Linear<CpuRuntime> {
    let (_, device) = cpu_setup();
    Linear::new(
        Tensor::<CpuRuntime>::from_slice(&vec![value; rows * cols], &[rows, cols], &device),
        None,
        false,
    )
}

fn tiny_mamba1(config: Mamba1Config) -> Mamba1<CpuRuntime> {
    let (_, device) = cpu_setup();
    let d_inner = config.d_inner();
    let in_proj = linear_with_value(config.in_proj_dim(), config.d_model, 0.0);
    let conv1d = Conv1d::new(
        Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; config.conv_channels() * config.d_conv],
            &[config.conv_channels(), 1, config.d_conv],
            &device,
        ),
        None,
        1,
        PaddingMode::Custom(config.d_conv - 1, 0, 0, 0),
        1,
        config.conv_channels(),
        false,
    );
    let x_proj = linear_with_value(config.x_proj_dim(), d_inner, 0.0);
    let dt_proj = linear_with_value(d_inner, d_inner, 0.0);
    let out_proj = linear_with_value(config.d_model, d_inner, 0.0);
    let weights = Mamba1Weights {
        in_proj,
        conv1d,
        x_proj,
        dt_proj,
        out_proj,
        a_log: Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; d_inner * config.d_state],
            &[d_inner, config.d_state],
            &device,
        ),
        d_param: if config.use_d {
            Some(Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; d_inner],
                &[d_inner],
                &device,
            ))
        } else {
            None
        },
    };
    Mamba1::new(config, weights, false)
}

fn assert_named_shape<R: Runtime>(params: &[(String, &Var<R>)], name: &str, shape: &[usize]) {
    let actual = params
        .iter()
        .find(|(param_name, _)| param_name == name)
        .map(|(_, param)| param.shape().to_vec())
        .unwrap_or_else(|| panic!("missing parameter {name}"));
    assert_eq!(actual, shape, "shape mismatch for {name}");
}

#[test]
fn test_mamba1_init_from_empty_varmap_shapes_and_ssm_defaults() {
    let (client, device) = cpu_setup();
    let config = Mamba1Config::new(4)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_d(true);
    let mut varmap = VarMap::<CpuRuntime>::new();
    let mut vb = VarBuilder::new(&mut varmap, &device);

    let mamba = Mamba1::init(&config, &mut vb, DType::F32, &client, true).unwrap();
    let params = mamba.named_parameters();
    let mut names: Vec<&str> = params.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "a_log",
            "conv1d.weight",
            "d_param",
            "dt_proj.weight",
            "in_proj.weight",
            "out_proj.weight",
            "x_proj.weight",
        ]
    );
    assert_named_shape(
        &params,
        "in_proj.weight",
        &[config.in_proj_dim(), config.d_model],
    );
    assert_named_shape(
        &params,
        "conv1d.weight",
        &[config.conv_channels(), 1, config.d_conv],
    );
    assert_named_shape(
        &params,
        "x_proj.weight",
        &[config.x_proj_dim(), config.d_inner()],
    );
    assert_named_shape(
        &params,
        "dt_proj.weight",
        &[config.d_inner(), config.d_inner()],
    );
    assert_named_shape(
        &params,
        "out_proj.weight",
        &[config.d_model, config.d_inner()],
    );
    assert_named_shape(&params, "a_log", &[config.d_inner(), config.d_state]);
    assert_named_shape(&params, "d_param", &[config.d_inner()]);

    let a_log: Vec<f32> = mamba.a_log.tensor().contiguous().unwrap().to_vec();
    assert!(a_log.iter().all(|&value| value == 0.0));
    let d_param: Vec<f32> = mamba
        .d_param
        .as_ref()
        .unwrap()
        .tensor()
        .contiguous()
        .unwrap()
        .to_vec();
    assert!(d_param.iter().all(|&value| value == 1.0));

    let mut strict_varmap = VarMap::<CpuRuntime>::new();
    let mut strict_vb = VarBuilder::new(&mut strict_varmap, &device);
    assert!(Mamba1::from_varbuilder(&config, &mut strict_vb, false).is_err());
}

#[test]
fn test_mamba1_forward_shape() {
    let (client, device) = cpu_setup();
    let config = Mamba1Config::new(4)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_d(false);
    let mamba = tiny_mamba1(config);
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 3, 4], &device),
        false,
    );

    let out = mamba.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 4]);
}

#[test]
fn test_mamba1_forward_invalid_input() {
    let (client, device) = cpu_setup();
    let config = Mamba1Config::new(4)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_d(false);
    let mamba = tiny_mamba1(config);

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
fn test_selective_scan_matches_f64_reference() {
    let (client, device) = cpu_setup();
    let config = Mamba1Config::new(2)
        .with_expand(1)
        .with_d_state(2)
        .with_dt_softplus(false)
        .with_use_d(true)
        .with_hidden_state_clamp(Some(30.0));
    let mut mamba = tiny_mamba1(config);
    mamba.d_param = Some(Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, -0.2], &[2], &device),
        false,
    ));

    let u_data = [1.0f32, 0.5, -0.25, 2.0, 0.75, -1.0];
    let delta_data = [0.2f32, 0.1, 0.3, 0.25, 0.15, 0.35];
    let a_data = [-0.5f32, -1.0, -0.25, -0.75];
    let b_data = [0.5f32, -0.2, 0.1, 0.3, -0.4, 0.25];
    let c_data = [1.0f32, 0.5, -0.25, 0.75, 0.6, -0.1];
    let d_data = [0.1f32, -0.2];

    let u = Var::new(
        Tensor::<CpuRuntime>::from_slice(&u_data, &[1, 3, 2], &device),
        false,
    );
    let delta = Var::new(
        Tensor::<CpuRuntime>::from_slice(&delta_data, &[1, 3, 2], &device),
        false,
    );
    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device),
        false,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 3, 2], &device),
        false,
    );
    let c = Var::new(
        Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 3, 2], &device),
        false,
    );

    let out = mamba
        .selective_scan_from_parts(&client, &u, &delta, &a, &b, &c, 1, 3)
        .unwrap();
    let data: Vec<f32> = out.tensor().to_vec();
    let expected = selective_scan_reference(
        &u_data,
        &delta_data,
        &a_data,
        &b_data,
        &c_data,
        Some(&d_data),
        Some(30.0),
        3,
        2,
        2,
    );

    for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual as f64 - expected).abs() < 2e-5,
            "idx={i}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn test_selective_scan_hidden_state_clamp_matches_f64_reference() {
    let (client, device) = cpu_setup();
    let config = Mamba1Config::new(1)
        .with_expand(1)
        .with_d_state(1)
        .with_dt_softplus(false)
        .with_use_d(false)
        .with_hidden_state_clamp(Some(1.0));
    let mamba = tiny_mamba1(config);

    let u_data = [10.0f32, 10.0];
    let delta_data = [1.0f32, 1.0];
    let a_data = [1.0f32];
    let b_data = [1.0f32, 1.0];
    let c_data = [1.0f32, 1.0];

    let u = Var::new(
        Tensor::<CpuRuntime>::from_slice(&u_data, &[1, 2, 1], &device),
        false,
    );
    let delta = Var::new(
        Tensor::<CpuRuntime>::from_slice(&delta_data, &[1, 2, 1], &device),
        false,
    );
    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&a_data, &[1, 1], &device),
        false,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 2, 1], &device),
        false,
    );
    let c = Var::new(
        Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 2, 1], &device),
        false,
    );

    let out = mamba
        .selective_scan_from_parts(&client, &u, &delta, &a, &b, &c, 1, 2)
        .unwrap();
    let data: Vec<f32> = out.tensor().to_vec();
    let expected = selective_scan_reference(
        &u_data,
        &delta_data,
        &a_data,
        &b_data,
        &c_data,
        None,
        Some(1.0),
        2,
        1,
        1,
    );

    for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual as f64 - expected).abs() < 1e-6,
            "idx={i}: actual={actual}, expected={expected}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn selective_scan_reference(
    u: &[f32],
    delta: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: Option<&[f32]>,
    clamp: Option<f64>,
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) -> Vec<f64> {
    let mut h = vec![0.0f64; d_inner * d_state];
    let mut out = Vec::with_capacity(seq_len * d_inner);

    for t in 0..seq_len {
        for dim in 0..d_inner {
            let u_t = u[t * d_inner + dim] as f64;
            let delta_t = delta[t * d_inner + dim] as f64;
            for state in 0..d_state {
                let idx = dim * d_state + state;
                let b_t = b[t * d_state + state] as f64;
                h[idx] = (delta_t * a[idx] as f64).exp() * h[idx] + delta_t * b_t * u_t;
                if let Some(limit) = clamp {
                    h[idx] = h[idx].clamp(-limit, limit);
                }
            }
        }

        for dim in 0..d_inner {
            let mut y = 0.0f64;
            for state in 0..d_state {
                y += h[dim * d_state + state] * c[t * d_state + state] as f64;
            }
            if let Some(d_param) = d {
                y += d_param[dim] as f64 * u[t * d_inner + dim] as f64;
            }
            out.push(y);
        }
    }

    out
}
