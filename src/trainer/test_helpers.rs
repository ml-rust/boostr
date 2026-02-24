//! Shared test utilities for trainer tests.

use crate::trainer::checkpoint::{CHECKPOINT_VERSION, TrainingState};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::collections::HashMap;

pub fn make_device() -> CpuDevice {
    CpuDevice::new()
}

pub fn make_training_state(step: u64) -> TrainingState {
    TrainingState {
        version: CHECKPOINT_VERSION,
        step,
        epoch: 0,
        learning_rate: 1e-3,
        metadata: HashMap::new(),
    }
}

pub fn make_model_state(device: &CpuDevice) -> HashMap<String, Tensor<CpuRuntime>> {
    let mut state = HashMap::new();
    state.insert(
        "layers.0.weight".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device),
    );
    state.insert(
        "layers.0.bias".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], device),
    );
    state
}

pub fn make_opt_state(device: &CpuDevice) -> HashMap<String, Tensor<CpuRuntime>> {
    let mut state = HashMap::new();
    state.insert(
        "layers.0.weight.m".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[0.01f32, 0.02, 0.03, 0.04], &[2, 2], device),
    );
    state.insert(
        "layers.0.weight.v".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[0.001f32, 0.002, 0.003, 0.004], &[2, 2], device),
    );
    state
}
