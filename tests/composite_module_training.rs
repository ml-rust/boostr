mod common;

use std::collections::HashSet;

use boostr::model::mamba::{
    Mamba1, Mamba1Config, Mamba1WeightsWithIds, Mamba3, Mamba3Config, Mamba3WeightsWithIds,
};
use boostr::nn::Module;
use boostr::trainer::{SimpleTrainer, TrainingConfig};
use common::{
    assert_ids, build_mamba2, build_mla, conv, cpu_setup, ids_from, linear, rebuild_mamba2,
    rebuild_mla, rms, trainable_map,
};
use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::{Tensor, TensorId};

#[test]
fn composite_modules_enumerate_all_trainable_parameters() {
    let (_client, device) = cpu_setup();
    let (mla, _mla_config, _mla_ids, mla_expected) = build_mla(&device);
    let mla_ids = ids_from(mla.trainable_parameters());
    assert_ids(&mla_ids, &mla_expected);
    assert_eq!(mla_ids.len(), 7);

    let (mamba2, _mamba2_config, mamba2_ids, mamba2_expected) = build_mamba2(&device);
    let mamba2_actual = ids_from(mamba2.trainable_parameters());
    assert_ids(&mamba2_actual, &mamba2_expected);
    assert_eq!(mamba2_actual.len(), 10);
    for raw_id in [mamba2_ids.a_log, mamba2_ids.dt_bias, mamba2_ids.d_param] {
        assert!(mamba2_actual.contains(&raw_id));
    }
    let mamba2_names: HashSet<String> = mamba2
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    for raw_name in ["a_log", "dt_bias", "d_param"] {
        assert!(mamba2_names.contains(raw_name));
    }

    let mamba1_config = Mamba1Config::new(4)
        .with_d_state(2)
        .with_expand(1)
        .with_d_conv(2)
        .with_use_d(true);
    let (m1_in, m1_in_ids) = linear(mamba1_config.in_proj_dim(), 4, true, 0.12, &device);
    let (m1_conv, m1_conv_ids) = conv(
        mamba1_config.conv_channels(),
        mamba1_config.d_conv,
        true,
        0.13,
        &device,
    );
    let (m1_x, m1_x_ids) = linear(
        mamba1_config.x_proj_dim(),
        mamba1_config.d_inner(),
        true,
        0.14,
        &device,
    );
    let (m1_dt, m1_dt_ids) = linear(
        mamba1_config.d_inner(),
        mamba1_config.d_inner(),
        true,
        0.15,
        &device,
    );
    let (m1_out, m1_out_ids) = linear(4, mamba1_config.d_inner(), true, 0.16, &device);
    let m1_a = Tensor::<CpuRuntime>::from_slice(&[-0.2f32; 4], &[mamba1_config.d_inner()], &device)
        .unwrap();
    let m1_a_id = m1_a.id();
    let m1_d = Tensor::<CpuRuntime>::from_slice(&[0.3f32; 4], &[mamba1_config.d_inner()], &device)
        .unwrap();
    let m1_d_id = m1_d.id();
    let mamba1 = Mamba1::with_ids(
        mamba1_config,
        Mamba1WeightsWithIds {
            in_proj: m1_in,
            conv1d: m1_conv,
            x_proj: m1_x,
            dt_proj: m1_dt,
            out_proj: m1_out,
            a_log: (m1_a, m1_a_id),
            d_param: Some((m1_d, m1_d_id)),
        },
        true,
    );
    let mamba1_expected = vec![
        m1_in_ids.weight,
        m1_in_ids.bias.expect("m1 in bias"),
        m1_conv_ids.weight,
        m1_conv_ids.bias.expect("m1 conv bias"),
        m1_x_ids.weight,
        m1_x_ids.bias.expect("m1 x bias"),
        m1_dt_ids.weight,
        m1_dt_ids.bias.expect("m1 dt bias"),
        m1_out_ids.weight,
        m1_out_ids.bias.expect("m1 out bias"),
        m1_a_id,
        m1_d_id,
    ];
    let mamba1_actual = ids_from(mamba1.trainable_parameters());
    assert_ids(&mamba1_actual, &mamba1_expected);
    assert_eq!(mamba1_actual.len(), 12);
    assert!(mamba1_actual.contains(&m1_a_id));
    assert!(mamba1_actual.contains(&m1_d_id));

    let mamba3_config = Mamba3Config::new(4)
        .with_nheads(1)
        .with_d_state(2)
        .with_expand(1)
        .with_complex_rope(true)
        .with_mimo_rank(2)
        .with_use_conv(true)
        .with_d_conv(2)
        .with_use_dt_bias(true)
        .with_use_d(true);
    let (m3_in, m3_in_ids) = linear(mamba3_config.proj_dim(), 4, true, 0.17, &device);
    let (m3_out, m3_out_ids) = linear(4, mamba3_config.d_inner(), true, 0.18, &device);
    let (m3_lambda, m3_lambda_ids) = linear(mamba3_config.nheads, 4, true, 0.19, &device);
    let (m3_theta, m3_theta_ids) = linear(
        mamba3_config.nheads * (mamba3_config.d_state / 2),
        4,
        true,
        0.20,
        &device,
    );
    let m3_b = Tensor::<CpuRuntime>::from_slice(
        &[0.1f32; 2],
        &[mamba3_config.nheads, mamba3_config.d_state],
        &device,
    )
    .unwrap();
    let m3_b_id = m3_b.id();
    let m3_c = Tensor::<CpuRuntime>::from_slice(
        &[0.2f32; 2],
        &[mamba3_config.nheads, mamba3_config.d_state],
        &device,
    )
    .unwrap();
    let m3_c_id = m3_c.id();
    let m3_dt =
        Tensor::<CpuRuntime>::from_slice(&[0.05f32], &[mamba3_config.nheads], &device).unwrap();
    let m3_dt_id = m3_dt.id();
    let m3_a =
        Tensor::<CpuRuntime>::from_slice(&[-0.4f32], &[mamba3_config.nheads], &device).unwrap();
    let m3_a_id = m3_a.id();
    let m3_d =
        Tensor::<CpuRuntime>::from_slice(&[0.3f32], &[mamba3_config.nheads], &device).unwrap();
    let m3_d_id = m3_d.id();
    let (m3_bc_norm, m3_bc_norm_id) = rms(mamba3_config.d_state, 1e-6, 0.21, &device);
    let (m3_norm, m3_norm_id) = rms(mamba3_config.d_inner(), 1e-6, 0.22, &device);
    let (m3_conv, m3_conv_ids) = conv(
        mamba3_config.conv_channels(),
        mamba3_config.d_conv,
        true,
        0.23,
        &device,
    );
    let (m3_up, m3_up_ids) = linear(
        mamba3_config.headdim * mamba3_config.mimo_rank,
        mamba3_config.headdim,
        true,
        0.24,
        &device,
    );
    let (m3_down, m3_down_ids) = linear(
        mamba3_config.headdim,
        mamba3_config.headdim * mamba3_config.mimo_rank,
        true,
        0.25,
        &device,
    );
    let mamba3 = Mamba3::with_ids(
        mamba3_config,
        Mamba3WeightsWithIds {
            in_proj: m3_in,
            out_proj: m3_out,
            lambda_proj: m3_lambda,
            theta_proj: Some(m3_theta),
            b_bias: (m3_b, m3_b_id),
            c_bias: (m3_c, m3_c_id),
            dt_bias: Some((m3_dt, m3_dt_id)),
            a_log: (m3_a, m3_a_id),
            d_param: Some((m3_d, m3_d_id)),
            bc_norm: m3_bc_norm,
            norm: m3_norm,
            conv1d: Some(m3_conv),
            mimo_x_up: Some(m3_up),
            mimo_x_down: Some(m3_down),
        },
        true,
    );
    let mamba3_expected = vec![
        m3_in_ids.weight,
        m3_in_ids.bias.expect("m3 in bias"),
        m3_out_ids.weight,
        m3_out_ids.bias.expect("m3 out bias"),
        m3_lambda_ids.weight,
        m3_lambda_ids.bias.expect("m3 lambda bias"),
        m3_theta_ids.weight,
        m3_theta_ids.bias.expect("m3 theta bias"),
        m3_b_id,
        m3_c_id,
        m3_dt_id,
        m3_a_id,
        m3_d_id,
        m3_bc_norm_id,
        m3_norm_id,
        m3_conv_ids.weight,
        m3_conv_ids.bias.expect("m3 conv bias"),
        m3_up_ids.weight,
        m3_up_ids.bias.expect("m3 up bias"),
        m3_down_ids.weight,
        m3_down_ids.bias.expect("m3 down bias"),
    ];
    let mamba3_actual = ids_from(mamba3.trainable_parameters());
    assert_ids(&mamba3_actual, &mamba3_expected);
    assert_eq!(mamba3_actual.len(), 21);
    for raw_id in [m3_b_id, m3_c_id, m3_dt_id, m3_a_id, m3_d_id] {
        assert!(mamba3_actual.contains(&raw_id));
    }
}

#[test]
fn composite_adamw_training_preserves_ids_and_state() {
    let (client, device) = cpu_setup();
    let mla_input = Tensor::<CpuRuntime>::from_slice(
        &[0.2f32, -0.1, 0.4, 0.3, -0.3, 0.5, -0.2, 0.1],
        &[1, 2, 4],
        &device,
    )
    .unwrap();
    let mla_target = Tensor::<CpuRuntime>::zeros(&[1, 2, 4], DType::F32, &device).unwrap();
    let (mla, mla_config, mla_ids, mla_expected_ids) = build_mla(&device);
    let mut mla_params = trainable_map(mla.trainable_parameters());
    assert_eq!(mla_params.len(), mla_expected_ids.len());

    let mamba_input = Tensor::<CpuRuntime>::from_slice(
        &[
            0.1f32, -0.2, 0.3, -0.4, 0.2, 0.1, -0.3, 0.4, -0.1, 0.3, 0.2, -0.2,
        ],
        &[1, 3, 4],
        &device,
    )
    .unwrap();
    let mamba_target = Tensor::<CpuRuntime>::zeros(&[1, 3, 4], DType::F32, &device).unwrap();
    let (mamba, mamba_config, mamba_ids, mamba_expected_ids) = build_mamba2(&device);
    let mut mamba_params = trainable_map(mamba.trainable_parameters());
    assert_eq!(mamba_params.len(), mamba_expected_ids.len());

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_weight_decay(0.0)
        .with_max_grad_norm(None);
    let mut mla_trainer = SimpleTrainer::<CpuRuntime>::new(config.clone()).expect("valid config");
    let mut mamba_trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");
    let (mut first_mla_loss, mut last_mla_loss) = (0.0f64, 0.0f64);
    let (mut first_mamba_loss, mut last_mamba_loss) = (0.0f64, 0.0f64);

    for step in 0..8 {
        let model = rebuild_mla(&mla_params, &mla_config, mla_ids, &device);
        assert_ids(&ids_from(model.trainable_parameters()), &mla_expected_ids);
        let x = Var::new(mla_input.clone(), false);
        let y = Var::new(mla_target.clone(), false);
        let pred = model.forward(&client, &x).expect("MLA forward");
        let diff = var_sub(&pred, &y, &client).expect("MLA diff");
        let sq = var_mul(&diff, &diff, &client).expect("MLA sq");
        let loss = var_mean(&sq, &[0, 1, 2], false, &client).expect("MLA loss");
        let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
        if step == 0 {
            first_mla_loss = loss_val;
        }
        last_mla_loss = loss_val;
        let grads = backward(&loss, &client).expect("MLA backward");
        for id in &mla_expected_ids {
            assert!(
                grads.get(*id).is_some(),
                "MLA gradient missing for stable id {id}"
            );
        }
        assert!(
            mla_trainer
                .step(&client, &mut mla_params, grads, loss_val)
                .expect("MLA optimizer step")
                .is_some()
        );
        assert_optimizer_state(
            mla_trainer.optimizer().state_len(),
            &mla_expected_ids,
            || mla_trainer.optimizer().state_ids().collect(),
        );
    }

    for step in 0..8 {
        let model = rebuild_mamba2(&mamba_params, &mamba_config, mamba_ids);
        assert_ids(&ids_from(model.trainable_parameters()), &mamba_expected_ids);
        let x = Var::new(mamba_input.clone(), false);
        let y = Var::new(mamba_target.clone(), false);
        let pred = model.forward(&client, &x).expect("Mamba2 forward");
        let diff = var_sub(&pred, &y, &client).expect("Mamba2 diff");
        let sq = var_mul(&diff, &diff, &client).expect("Mamba2 sq");
        let loss = var_mean(&sq, &[0, 1, 2], false, &client).expect("Mamba2 loss");
        let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
        if step == 0 {
            first_mamba_loss = loss_val;
        }
        last_mamba_loss = loss_val;
        let grads = backward(&loss, &client).expect("Mamba2 backward");
        for id in &mamba_expected_ids {
            assert!(
                grads.get(*id).is_some(),
                "Mamba2 gradient missing for stable id {id}"
            );
        }
        assert!(
            mamba_trainer
                .step(&client, &mut mamba_params, grads, loss_val)
                .expect("Mamba2 optimizer step")
                .is_some()
        );
        assert_optimizer_state(
            mamba_trainer.optimizer().state_len(),
            &mamba_expected_ids,
            || mamba_trainer.optimizer().state_ids().collect(),
        );
    }

    assert!(
        last_mla_loss < first_mla_loss,
        "MLA loss should decrease: first={first_mla_loss} last={last_mla_loss}"
    );
    assert!(
        last_mamba_loss < first_mamba_loss,
        "Mamba2 loss should decrease: first={first_mamba_loss} last={last_mamba_loss}"
    );
}

fn assert_optimizer_state<F>(state_len: usize, expected: &[TensorId], state_ids: F)
where
    F: FnOnce() -> HashSet<TensorId>,
{
    assert_eq!(state_len, expected.len());
    assert_eq!(
        state_ids(),
        expected.iter().copied().collect::<HashSet<_>>()
    );
}
