use std::collections::HashMap;

use boostr::nn::{Conv1d, Embedding, LayerNorm, Linear, RmsNorm};
use boostr::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
use numr::ops::PaddingMode;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::{Tensor, TensorId};

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

fn parameter_tensor_map(
    params: Vec<(TensorId, &Var<CpuRuntime>)>,
) -> HashMap<TensorId, Tensor<CpuRuntime>> {
    params
        .into_iter()
        .map(|(id, var)| (id, var.tensor().clone()))
        .collect()
}

#[test]
fn listed_nn_modules_preserve_explicit_parameter_ids() {
    let (_client, device) = cpu_setup();

    let linear_w = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
    let linear_w_id = linear_w.id();
    let linear_b = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[2], &device);
    let linear_b_id = linear_b.id();
    let linear = Linear::with_ids(
        linear_w.clone(),
        linear_w_id,
        Some((linear_b.clone(), linear_b_id)),
        true,
    );
    assert_eq!(
        linear
            .parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        vec![linear_w_id, linear_b_id]
    );

    let embedding_w = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, 4], &device);
    let embedding_w_id = embedding_w.id();
    let embedding = Embedding::with_id(embedding_w.clone(), embedding_w_id, true);
    assert_eq!(embedding.parameters()[0].0, embedding_w_id);

    let layernorm_w = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
    let layernorm_w_id = layernorm_w.id();
    let layernorm_b = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
    let layernorm_b_id = layernorm_b.id();
    let layernorm = LayerNorm::with_ids(
        layernorm_w.clone(),
        layernorm_w_id,
        layernorm_b.clone(),
        layernorm_b_id,
        1e-5,
        true,
    );
    assert_eq!(
        layernorm
            .parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        vec![layernorm_w_id, layernorm_b_id]
    );

    let rms_w = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
    let rms_w_id = rms_w.id();
    let rms = RmsNorm::with_id(rms_w.clone(), rms_w_id, 1e-5, true);
    assert_eq!(rms.parameters()[0].0, rms_w_id);

    let conv_w = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 3], &[1, 1, 3], &device);
    let conv_w_id = conv_w.id();
    let conv_b = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);
    let conv_b_id = conv_b.id();
    let conv = Conv1d::with_ids(
        conv_w.clone(),
        conv_w_id,
        Some((conv_b.clone(), conv_b_id)),
        1,
        PaddingMode::Valid,
        1,
        1,
        true,
    );
    assert_eq!(
        conv.parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        vec![conv_w_id, conv_b_id]
    );
}

#[test]
fn nn_module_adamw_loop_preserves_state_and_converges() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.02)
        .with_weight_decay(0.0)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1], &device);
    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1], &device);

    let weight_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1], &device);
    let weight_id = weight_init.id();
    let bias_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);
    let bias_id = bias_init.id();

    let initial_model = Linear::with_ids(weight_init, weight_id, Some((bias_init, bias_id)), true);
    let expected_ids: Vec<TensorId> = initial_model
        .trainable_parameters()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(expected_ids, vec![weight_id, bias_id]);

    let mut params = parameter_tensor_map(initial_model.trainable_parameters());
    assert_eq!(params.len(), expected_ids.len());

    let mut first_loss = 0.0f64;
    let mut previous_loss = f64::INFINITY;
    let mut last_loss = 0.0f64;

    for step in 0..20 {
        let weight = params
            .get(&weight_id)
            .expect("weight param keyed by stable TensorId")
            .clone();
        let bias = params
            .get(&bias_id)
            .expect("bias param keyed by stable TensorId")
            .clone();
        let model = Linear::with_ids(weight, weight_id, Some((bias, bias_id)), true);

        let ids: Vec<TensorId> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            ids, expected_ids,
            "parameter TensorIds must stay stable after rebuild at step {step}"
        );

        let x = Var::new(input.clone(), false);
        let y = Var::new(target.clone(), false);
        let pred = model.forward(&client, &x).expect("linear forward");
        let diff = var_sub(&pred, &y, &client).expect("var_sub");
        let sq = var_mul(&diff, &diff, &client).expect("var_mul");
        let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
        let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;

        if step == 0 {
            first_loss = loss_val;
        } else {
            assert!(
                loss_val <= previous_loss + 1e-7,
                "loss must decrease monotonically: step={step} previous={previous_loss} current={loss_val}"
            );
        }
        previous_loss = loss_val;
        last_loss = loss_val;

        let grads = backward(&loss, &client).expect("backward");
        for id in &expected_ids {
            assert!(
                grads.get(*id).is_some(),
                "gradient must be keyed by stable TensorId {id}"
            );
        }

        let metrics = trainer
            .step(&client, &mut params, grads, loss_val)
            .expect("trainer step");
        assert!(metrics.is_some());

        assert_eq!(
            trainer.optimizer().state_len(),
            expected_ids.len(),
            "AdamW should keep exactly one moment state per logical parameter; changed IDs would orphan old states"
        );
        for id in &expected_ids {
            assert!(
                trainer.optimizer().has_state(*id),
                "AdamW moment state must persist for stable TensorId {id}"
            );
        }
    }

    assert!(
        last_loss < first_loss * 0.2,
        "loss should decrease significantly: first={first_loss} last={last_loss}"
    );
}
