use std::collections::HashMap;

use boostr::optimizer::LrSchedule;
use boostr::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

#[test]
fn test_trainer_basic_step() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target.clone(), false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
    let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
    let grads = backward(&loss, &client).expect("backward");

    let metrics = trainer
        .step(&client, &mut params, grads, loss_val)
        .expect("step");
    assert!(metrics.is_some());

    let m = metrics.unwrap();
    assert_eq!(m.step, 1);
    assert!(m.loss > 0.0);
}

#[test]
fn test_trainer_with_grad_accum() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_grad_accum_steps(2)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    // First micro-batch: should return None (accumulating)
    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target.clone(), false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0], false, &client).expect("var_mean");
    let grads = backward(&loss, &client).expect("backward");

    let result = trainer
        .step(&client, &mut params, grads, 1.0)
        .expect("step");
    assert!(result.is_none());

    // Second micro-batch: should return Some (step happens)
    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target.clone(), false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0], false, &client).expect("var_mean");
    let grads = backward(&loss, &client).expect("backward");

    let result = trainer
        .step(&client, &mut params, grads, 1.0)
        .expect("step");
    assert!(result.is_some());
    assert_eq!(result.unwrap().step, 1);
}

#[test]
fn test_trainer_convergence() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.1)
        .with_weight_decay(0.0)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    let mut first_loss = 0.0;
    let mut last_loss = 0.0;

    for i in 0..30 {
        let w = Var::with_id(params[&w_id].clone(), w_id, true);
        let t = Var::new(target.clone(), false);
        let diff = var_sub(&w, &t, &client).expect("var_sub");
        let sq = var_mul(&diff, &diff, &client).expect("var_mul");
        let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
        let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
        let grads = backward(&loss, &client).expect("backward");

        if i == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        trainer
            .step(&client, &mut params, grads, loss_val)
            .expect("step");
    }

    assert!(
        last_loss < first_loss * 0.1,
        "loss should decrease: first={first_loss} last={last_loss}"
    );
}

#[test]
fn test_trainer_with_lr_schedule() {
    let config = TrainingConfig::default().with_lr(0.01);
    let trainer = SimpleTrainer::<CpuRuntime>::new(config)
        .expect("valid config")
        .with_lr_schedule(LrSchedule::CosineAnnealing {
            base_lr: 0.01,
            min_lr: 0.001,
            warmup_steps: 10,
            total_steps: 100,
        });

    assert_eq!(trainer.global_step(), 0);
}

#[test]
fn test_trainer_with_grad_clipping() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_max_grad_norm(Some(0.1));
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target, false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
    let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
    let grads = backward(&loss, &client).expect("backward");

    let metrics = trainer
        .step(&client, &mut params, grads, loss_val)
        .expect("step");
    assert!(metrics.is_some());

    let m = metrics.unwrap();
    assert!(m.grad_norm.is_some());
    assert!(m.grad_norm.unwrap() > 0.0);
}
