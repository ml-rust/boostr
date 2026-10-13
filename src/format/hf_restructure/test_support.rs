//! Shared fixtures for the checkpoint-restructure unit tests: a config
//! from JSON, a flat row dump, and a row-tagged tensor.

use crate::model::config::UniversalConfig;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(super) fn config(json: &str) -> UniversalConfig {
    let hf: crate::model::config::HuggingFaceConfig = serde_json::from_str(json).unwrap();
    hf.to_universal().unwrap()
}

pub(super) fn rows(t: &Tensor<CpuRuntime>) -> Vec<f32> {
    t.contiguous().unwrap().to_vec()
}

/// Row `r` of a `[n, 1]` tensor holds `r`.
pub(super) fn tagged(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let v: Vec<f32> = (0..n).map(|r| r as f32).collect();
    Tensor::from_slice(&v, &[n, 1], device).unwrap()
}
