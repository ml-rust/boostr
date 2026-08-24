use crate::model::mamba::mamba2::config::Mamba2Config;
use crate::model::mamba::mamba2::layer::{Mamba2, Mamba2Weights};
use crate::nn::{Conv1d, Linear, Module, VarBuilder, VarMap};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::{Runtime, cpu::CpuRuntime};
use numr::tensor::Tensor;

/// Same tiny layer as [`tiny_mamba2`] but with softplus enabled and an explicit
/// `dt_bias` filled with `bias_value`, for exercising the dt ordering.
fn mamba2_with_dt_bias(bias_value: f32) -> (Mamba2<CpuRuntime>, Mamba2Config) {
    let (_, device) = cpu_setup();
    let config = Mamba2Config::new(8)
        .with_nheads(1)
        .with_d_state(4)
        .with_expand(2)
        .with_dt_softplus(true)
        .with_use_dt_bias(true)
        .with_use_d(false);

    let d_inner = config.d_inner();
    let conv_channels = config.conv_channels();
    let proj_dim = config.proj_dim();

    let in_proj = Linear::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.01f32; 328], &[proj_dim, 8], &device).unwrap(),
        None,
        false,
    );
    let conv1d = Conv1d::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device)
            .unwrap(),
        None,
        1,
        PaddingMode::Custom(3, 0, 0, 0),
        1,
        conv_channels,
        false,
    );
    let out_proj = Linear::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.01f32; 128], &[8, d_inner], &device).unwrap(),
        None,
        false,
    );
    let a_log =
        Tensor::<CpuRuntime>::try_from_slice(&[-0.5f32], &[config.nheads], &device).unwrap();
    let dt_bias =
        Tensor::<CpuRuntime>::try_from_slice(&[bias_value], &[config.nheads], &device).unwrap();

    let weights = Mamba2Weights {
        in_proj,
        conv1d,
        out_proj,
        a_log,
        dt_bias: Some(dt_bias),
        d_param: None,
        norm: None,
    };
    let mamba = Mamba2::new(config.clone(), weights, false);
    (mamba, config)
}

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
        Tensor::<CpuRuntime>::try_from_slice(&[0.01f32; 328], &[proj_dim, 8], &device).unwrap(),
        None,
        false,
    );
    let conv1d = Conv1d::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device)
            .unwrap(),
        None,
        1,
        PaddingMode::Custom(3, 0, 0, 0),
        1,
        conv_channels,
        false,
    );
    let out_proj = Linear::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.01f32; 128], &[8, d_inner], &device).unwrap(),
        None,
        false,
    );
    let a_log =
        Tensor::<CpuRuntime>::try_from_slice(&[-0.5f32], &[config.nheads], &device).unwrap();

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

fn assert_named_shape<R: Runtime>(params: &[(String, &Var<R>)], name: &str, shape: &[usize]) {
    let actual = params
        .iter()
        .find(|(param_name, _)| param_name == name)
        .map(|(_, param)| param.shape().to_vec())
        .unwrap_or_else(|| panic!("missing parameter {name}"));
    assert_eq!(actual, shape, "shape mismatch for {name}");
}

#[test]
fn test_mamba2_init_from_empty_varmap_shapes_and_ssm_defaults() {
    let (client, device) = cpu_setup();
    let config = Mamba2Config::new(8)
        .with_nheads(1)
        .with_d_state(4)
        .with_expand(2)
        .with_dt_softplus(false)
        .with_use_dt_bias(true)
        .with_use_d(true);
    let mut varmap = VarMap::<CpuRuntime>::new();
    let mut vb = VarBuilder::new(&mut varmap, &device);

    let mamba = Mamba2::init(&config, &mut vb, DType::F32, &client, true).unwrap();
    let params = mamba.named_parameters();
    let mut names: Vec<&str> = params.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "a_log",
            "conv1d.weight",
            "d_param",
            "dt_bias",
            "in_proj.weight",
            "out_proj.weight",
        ]
    );
    assert_named_shape(
        &params,
        "in_proj.weight",
        &[config.proj_dim(), config.d_model],
    );
    assert_named_shape(
        &params,
        "conv1d.weight",
        &[config.conv_channels(), 1, config.d_conv],
    );
    assert_named_shape(
        &params,
        "out_proj.weight",
        &[config.d_model, config.d_inner()],
    );
    assert_named_shape(&params, "a_log", &[config.nheads]);
    assert_named_shape(&params, "dt_bias", &[config.nheads]);
    assert_named_shape(&params, "d_param", &[config.nheads]);

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
    assert!(Mamba2::from_varbuilder(&config, &mut strict_vb, false).is_err());
}

#[test]
fn test_mamba2_forward_shape() {
    let (client, device) = cpu_setup();
    let (mamba, _) = tiny_mamba2();

    let x = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.1f32; 32], &[1, 4, 8], &device).unwrap(),
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
        Tensor::<CpuRuntime>::try_from_slice(&[0.1f32; 8], &[1, 8], &device).unwrap(),
        false,
    );
    assert!(mamba.forward(&client, &x_2d).is_err());

    // Wrong d_model should fail
    let x_wrong = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.1f32; 12], &[1, 4, 3], &device).unwrap(),
        false,
    );
    assert!(mamba.forward(&client, &x_wrong).is_err());
}

#[test]
fn test_mamba2_model_config() {
    let config = crate::model::config::UniversalConfig {
        model_type: "mamba2".into(),
        vocab_size: 1000,
        hidden_size: 64,
        num_layers: 2,
        max_seq_len: 512,
        intermediate_size: None,
        rms_norm_eps: 1e-5,
        attention: None,
        ssm: Some(crate::model::config::SsmConfig {
            variant: "mamba2".into(),
            state_size: 16,
            num_heads: 2,
            head_dim: 64,
            expand: 2,
            conv_kernel: 4,
            chunk_size: 64,
            n_groups: 1,
            complex_rope: None,
            mimo_rank: None,
            use_conv: None,
        }),
        moe: None,
        hybrid_layers: None,
        tie_word_embeddings: false,
        grow_vocab: false,
        vision: None,
        audio: None,
    };
    let mamba_config = Mamba2Config::from_universal(&config).unwrap();
    assert_eq!(mamba_config.d_model, 64);
    assert_eq!(mamba_config.nheads, 2);
    assert_eq!(mamba_config.d_state, 16);
}

/// `dt_bias` must be added INSIDE softplus: `softplus(dt + bias)`.
///
/// Regression: this computed `softplus(dt) + bias`. With the default zero-init
/// bias the two are identical, so the bug is invisible until the bias trains
/// away from zero — at which point a sufficiently negative bias makes dt
/// negative, flipping the sign of the decay exponent `exp(dt * A)` so the
/// recurrence diverges instead of decaying.
///
/// A strongly negative bias separates the two orderings:
///   softplus(dt + bias) > 0 always
///   softplus(dt) + bias < 0 for bias below -softplus(dt)
#[test]
fn test_mamba2_dt_bias_is_applied_inside_softplus() {
    use numr::autograd::{var_add, var_softplus};

    let (client, device) = cpu_setup();

    // dt values around zero => softplus(dt) ~ 0.69; a -5.0 bias flips the sign
    // under the WRONG ordering but never under the correct one.
    let dt = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.25, -0.25, 0.5], &[4], &device).unwrap(),
        false,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[-5.0f32; 4], &[4], &device).unwrap(),
        false,
    );

    // Correct: bias inside.
    let inside = var_softplus(&var_add(&dt, &bias, &client).unwrap(), &client).unwrap();
    let inside_vals: Vec<f32> = inside.tensor().contiguous().unwrap().to_vec();

    // Wrong: bias outside.
    let outside = var_add(&var_softplus(&dt, &client).unwrap(), &bias, &client).unwrap();
    let outside_vals: Vec<f32> = outside.tensor().contiguous().unwrap().to_vec();

    assert!(
        inside_vals.iter().all(|v| *v > 0.0),
        "softplus(dt + bias) must stay positive, got {inside_vals:?}"
    );
    assert!(
        outside_vals.iter().all(|v| *v < 0.0),
        "test setup is degenerate: the wrong ordering should go negative here, got {outside_vals:?}"
    );

    // The arithmetic above only pins the semantics; now prove the LAYER uses it.
    //
    // Compare a strongly negative bias against a zero bias. dt scales the SSM
    // input term, so the two orderings move the output in OPPOSITE directions:
    //   correct  softplus(dt_raw - 5) ~= 0.007  -> much SMALLER than softplus(dt_raw) ~= 0.69
    //   wrong    softplus(dt_raw) - 5 ~= -4.31  -> |dt| much LARGER, and the decay
    //                                              exponent dt*A flips sign
    // Asserting the direction is robust; asserting a magnitude threshold is not,
    // because these tiny fixture weights never actually overflow.
    let magnitude = |bias: f32| -> f32 {
        let (mamba, _) = mamba2_with_dt_bias(bias);
        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.05f32; 8 * 6], &[1, 6, 8], &device).unwrap(),
            false,
        );
        let out = mamba.forward(&client, &x).expect("forward must succeed");
        let vals: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "dt_bias={bias} produced non-finite output"
        );
        vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    };

    let neutral = magnitude(0.0);
    let suppressed = magnitude(-5.0);
    assert!(
        neutral > 0.0,
        "test setup is degenerate: zero-bias output is exactly zero"
    );
    assert!(
        suppressed < neutral * 0.5,
        "a strongly negative dt_bias must SHRINK the output (dt -> 0); \
         got {suppressed} vs {neutral} at zero bias — dt_bias is being added \
         outside softplus"
    );
}
