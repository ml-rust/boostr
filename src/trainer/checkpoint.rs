//! Checkpoint save/load for training resumption.
//!
//! A checkpoint consists of:
//! - `model.safetensors` — model parameters
//! - `optimizer.safetensors` — optimizer state (m/v tensors for AdamW)
//! - `training_state.json` — step, epoch, lr, config, version

use crate::error::{Error, Result};
use crate::format::safetensors::{SafeTensors, save_safetensors};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Current checkpoint format version.
pub const CHECKPOINT_VERSION: u32 = 2;

fn default_version() -> u32 {
    1
}

/// Loaded checkpoint data: (model_state, optimizer_state, training_state).
pub type CheckpointData<R> = (
    HashMap<String, Tensor<R>>,
    Option<HashMap<String, Tensor<R>>>,
    TrainingState,
);

/// Training metadata saved alongside model/optimizer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Checkpoint format version. Defaults to 1 for old checkpoints without this field.
    #[serde(default = "default_version")]
    pub version: u32,
    pub step: u64,
    #[serde(default)]
    pub epoch: u64,
    #[serde(default)]
    pub learning_rate: f64,
    /// Arbitrary key-value metadata (loss history, config hash, etc.)
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Save a checkpoint to a directory.
///
/// Creates:
/// - `{dir}/model.safetensors`
/// - `{dir}/optimizer.safetensors` (if optimizer_state is provided)
/// - `{dir}/training_state.json`
pub fn save_checkpoint<P: AsRef<Path>>(
    dir: P,
    model_state: &HashMap<String, Tensor<CpuRuntime>>,
    optimizer_state: Option<&HashMap<String, Tensor<CpuRuntime>>>,
    training_state: &TrainingState,
) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| Error::TrainingError {
        reason: format!("failed to create checkpoint dir: {e}"),
    })?;

    // Model weights
    save_safetensors(dir.join("model.safetensors"), model_state, None)?;

    // Optimizer state
    if let Some(opt_state) = optimizer_state {
        if !opt_state.is_empty() {
            save_safetensors(dir.join("optimizer.safetensors"), opt_state, None)?;
        }
    }

    // Training metadata — always write current version
    let mut state = training_state.clone();
    state.version = CHECKPOINT_VERSION;
    let json = serde_json::to_string_pretty(&state).map_err(|e| Error::TrainingError {
        reason: format!("failed to serialize training state: {e}"),
    })?;
    std::fs::write(dir.join("training_state.json"), json).map_err(|e| Error::TrainingError {
        reason: format!("failed to write training state: {e}"),
    })?;

    Ok(())
}

/// Load a checkpoint from a directory.
///
/// Returns (model_state, optimizer_state, training_state).
/// optimizer_state is None if `optimizer.safetensors` doesn't exist.
///
/// Generic over runtime — loads directly to the target device (CPU, CUDA, etc.).
pub fn load_checkpoint<R: Runtime<DType = DType>, P: AsRef<Path>>(
    dir: P,
    device: &R::Device,
) -> Result<CheckpointData<R>> {
    let dir = dir.as_ref();

    // Model weights
    let model_path = dir.join("model.safetensors");
    let mut st = SafeTensors::open(&model_path)?;
    let model_state = st.load_all::<R>(device)?;

    // Optimizer state (optional)
    let opt_path = dir.join("optimizer.safetensors");
    let optimizer_state = if opt_path.exists() {
        let mut opt_st = SafeTensors::open(&opt_path)?;
        Some(opt_st.load_all::<R>(device)?)
    } else {
        None
    };

    // Training metadata
    let state_path = dir.join("training_state.json");
    let json = std::fs::read_to_string(&state_path).map_err(|e| Error::TrainingError {
        reason: format!("failed to read training state: {e}"),
    })?;
    let training_state: TrainingState =
        serde_json::from_str(&json).map_err(|e| Error::TrainingError {
            reason: format!("failed to parse training state: {e}"),
        })?;

    Ok((model_state, optimizer_state, training_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::test_helpers::*;
    use numr::runtime::cpu::CpuDevice;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load_checkpoint() {
        let dir = TempDir::new().unwrap();
        let device = make_device();
        let model_state = make_model_state(&device);
        let opt_state = make_opt_state(&device);
        let training_state = TrainingState {
            version: CHECKPOINT_VERSION,
            step: 1000,
            epoch: 2,
            learning_rate: 3e-4,
            metadata: HashMap::new(),
        };

        save_checkpoint(dir.path(), &model_state, Some(&opt_state), &training_state).unwrap();

        assert!(dir.path().join("model.safetensors").exists());
        assert!(dir.path().join("optimizer.safetensors").exists());
        assert!(dir.path().join("training_state.json").exists());

        let (loaded_model, loaded_opt, loaded_state) =
            load_checkpoint::<CpuRuntime, _>(dir.path(), &device).unwrap();

        assert_eq!(loaded_model.len(), 2);
        let w = &loaded_model["layers.0.weight"];
        assert_eq!(w.shape(), &[2, 2]);
        let data: Vec<f32> = w.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6);

        let opt = loaded_opt.unwrap();
        assert_eq!(opt.len(), 2);
        let m = &opt["layers.0.weight.m"];
        let m_data: Vec<f32> = m.to_vec();
        assert!((m_data[0] - 0.01).abs() < 1e-6);

        assert_eq!(loaded_state.step, 1000);
        assert_eq!(loaded_state.epoch, 2);
        assert!((loaded_state.learning_rate - 3e-4).abs() < 1e-10);
    }

    #[test]
    fn test_checkpoint_without_optimizer() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        let mut model_state = HashMap::new();
        model_state.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
        );

        let training_state = make_training_state(42);

        save_checkpoint(dir.path(), &model_state, None, &training_state).unwrap();
        assert!(!dir.path().join("optimizer.safetensors").exists());

        let (_, loaded_opt, loaded_state) =
            load_checkpoint::<CpuRuntime, _>(dir.path(), &device).unwrap();
        assert!(loaded_opt.is_none());
        assert_eq!(loaded_state.step, 42);
    }

    #[test]
    fn test_version_round_trip() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        let mut model_state = HashMap::new();
        model_state.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
        );
        let state = make_training_state(10);

        save_checkpoint(dir.path(), &model_state, None, &state).unwrap();

        let (_, _, loaded) = load_checkpoint::<CpuRuntime, _>(dir.path(), &device).unwrap();
        assert_eq!(loaded.version, CHECKPOINT_VERSION);
    }

    #[test]
    fn test_version_backward_compat() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        // Create a v1-style checkpoint (no version field)
        let mut model_state = HashMap::new();
        model_state.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
        );
        save_safetensors(dir.path().join("model.safetensors"), &model_state, None).unwrap();

        // Write training state JSON without version field (simulating v1)
        let json = r#"{"step": 100, "epoch": 5, "learning_rate": 0.001, "metadata": {}}"#;
        std::fs::write(dir.path().join("training_state.json"), json).unwrap();

        let (_, _, loaded) = load_checkpoint::<CpuRuntime, _>(dir.path(), &device).unwrap();
        assert_eq!(loaded.version, 1); // defaults to 1
        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.epoch, 5);
    }
}
