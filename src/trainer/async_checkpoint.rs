//! Async checkpoint writer — snapshots tensors to CPU, writes in a background thread.

use crate::error::{Error, Result};
use crate::trainer::checkpoint::{TrainingState, save_checkpoint};
use crate::trainer::distributed_checkpoint::{ShardingConfig, save_distributed_checkpoint};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Type alias for reconstructed checkpoint state (model tensors, optional optimizer tensors).
type ReconstructedState = (
    HashMap<String, Tensor<CpuRuntime>>,
    Option<HashMap<String, Tensor<CpuRuntime>>>,
);

/// Snapshot of tensor data for async writing.
/// Stores raw bytes + shape + dtype so we can reconstruct CPU tensors in the background thread.
struct TensorSnapshot {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

impl TensorSnapshot {
    /// Snapshot a tensor by copying its data to host memory.
    fn from_tensor<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> Result<Self> {
        Ok(Self {
            bytes: tensor.to_bytes().map_err(|e| Error::TrainingError {
                reason: format!("failed to snapshot tensor: {e}"),
            })?,
            shape: tensor.shape().to_vec(),
            dtype: tensor.dtype(),
        })
    }

    /// Reconstruct as a CPU tensor.
    fn to_cpu_tensor(&self) -> Result<Tensor<CpuRuntime>> {
        let device = CpuDevice::new();
        Tensor::<CpuRuntime>::try_from_bytes(&self.bytes, &self.shape, self.dtype, &device).map_err(
            |e| Error::TrainingError {
                reason: format!("failed to reconstruct tensor from snapshot: {e}"),
            },
        )
    }
}

/// Snapshot an entire state dict for async writing.
fn snapshot_state<R: Runtime<DType = DType>>(
    state: &HashMap<String, Tensor<R>>,
) -> Result<HashMap<String, TensorSnapshot>> {
    state
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), TensorSnapshot::from_tensor(tensor)?)))
        .collect()
}

/// Reconstruct CPU tensors from snapshots.
fn reconstruct_state(
    snapshots: &HashMap<String, TensorSnapshot>,
) -> Result<HashMap<String, Tensor<CpuRuntime>>> {
    snapshots
        .iter()
        .map(|(name, snap)| Ok((name.clone(), snap.to_cpu_tensor()?)))
        .collect()
}

/// Snapshotted state ready for background writing.
struct StateSnapshots {
    model: HashMap<String, TensorSnapshot>,
    optimizer: Option<HashMap<String, TensorSnapshot>>,
    training: TrainingState,
    dir: PathBuf,
}

impl StateSnapshots {
    /// Snapshot model and optimizer tensors, clone training state.
    fn capture<R: Runtime<DType = DType>, P: AsRef<Path>>(
        dir: P,
        model_state: &HashMap<String, Tensor<R>>,
        optimizer_state: Option<&HashMap<String, Tensor<R>>>,
        training_state: &TrainingState,
    ) -> Result<Self> {
        Ok(Self {
            model: snapshot_state(model_state)?,
            optimizer: match optimizer_state {
                Some(opt) => Some(snapshot_state(opt)?),
                None => None,
            },
            training: training_state.clone(),
            dir: dir.as_ref().to_path_buf(),
        })
    }

    /// Reconstruct CPU tensors from snapshots and return them with the training state.
    fn reconstruct(&self) -> Result<ReconstructedState> {
        let model_cpu = reconstruct_state(&self.model)?;
        let opt_cpu = match &self.optimizer {
            Some(snap) => Some(reconstruct_state(snap)?),
            None => None,
        };
        Ok((model_cpu, opt_cpu))
    }
}

/// Async checkpoint writer that snapshots tensor data and writes in a background thread.
///
/// Only one checkpoint write can be in-flight at a time. Starting a new save waits for
/// any previous write to complete first.
pub struct AsyncCheckpointer {
    handle: Option<std::thread::JoinHandle<Result<()>>>,
}

impl AsyncCheckpointer {
    /// Create a new async checkpointer.
    pub fn new() -> Self {
        Self { handle: None }
    }

    /// Wait for any in-progress checkpoint write to finish.
    pub fn wait(&mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|e| Error::TrainingError {
                reason: format!("checkpoint writer thread panicked: {e:?}"),
            })??;
        }
        Ok(())
    }

    /// Snapshot tensor state and write checkpoint in a background thread.
    ///
    /// If a previous checkpoint is still writing, waits for it first.
    /// Tensors are copied to CPU memory (snapshot) before spawning the writer thread,
    /// so the caller can safely continue modifying GPU tensors after this returns.
    pub fn save<R, P>(
        &mut self,
        dir: P,
        model_state: &HashMap<String, Tensor<R>>,
        optimizer_state: Option<&HashMap<String, Tensor<R>>>,
        training_state: &TrainingState,
    ) -> Result<()>
    where
        R: Runtime<DType = DType>,
        P: AsRef<Path>,
    {
        self.wait()?;

        let snap = StateSnapshots::capture(dir, model_state, optimizer_state, training_state)?;

        self.handle = Some(std::thread::spawn(move || {
            let (model_cpu, opt_cpu) = snap.reconstruct()?;
            save_checkpoint(&snap.dir, &model_cpu, opt_cpu.as_ref(), &snap.training)
        }));

        Ok(())
    }

    /// Snapshot tensor state and write a distributed checkpoint in a background thread.
    ///
    /// Combines async writing with distributed (per-rank) checkpointing.
    #[allow(clippy::too_many_arguments)]
    pub fn save_distributed<R, P>(
        &mut self,
        dir: P,
        rank: usize,
        world_size: usize,
        model_state: &HashMap<String, Tensor<R>>,
        optimizer_state: Option<&HashMap<String, Tensor<R>>>,
        training_state: &TrainingState,
        sharding: ShardingConfig,
    ) -> Result<()>
    where
        R: Runtime<DType = DType>,
        P: AsRef<Path>,
    {
        self.wait()?;

        let snap = StateSnapshots::capture(dir, model_state, optimizer_state, training_state)?;

        self.handle = Some(std::thread::spawn(move || {
            let (model_cpu, opt_cpu) = snap.reconstruct()?;
            save_distributed_checkpoint(
                &snap.dir,
                rank,
                world_size,
                &model_cpu,
                opt_cpu.as_ref(),
                &snap.training,
                sharding,
            )
        }));

        Ok(())
    }
}

impl Default for AsyncCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AsyncCheckpointer {
    fn drop(&mut self) {
        // Best-effort: wait for in-flight write so we don't lose a checkpoint.
        let _ = self.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::checkpoint::load_checkpoint;
    use crate::trainer::test_helpers::*;
    use numr::runtime::cpu::CpuRuntime;
    use tempfile::TempDir;

    #[test]
    fn test_async_save() {
        let dir = TempDir::new().unwrap();
        let device = make_device();
        let model_state = make_model_state(&device);
        let state = make_training_state(50);

        let mut checkpointer = AsyncCheckpointer::new();
        checkpointer
            .save(dir.path(), &model_state, None, &state)
            .unwrap();
        checkpointer.wait().unwrap();

        assert!(dir.path().join("model.safetensors").exists());
        assert!(dir.path().join("training_state.json").exists());

        let (loaded_model, _, loaded_state) =
            load_checkpoint::<CpuRuntime, _>(dir.path(), &device).unwrap();
        assert_eq!(loaded_model.len(), 2);
        assert_eq!(loaded_state.step, 50);

        let w: Vec<f32> = loaded_model["layers.0.weight"].to_vec();
        assert!((w[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_async_double_save() {
        let dir1 = TempDir::new().unwrap();
        let dir2 = TempDir::new().unwrap();
        let device = make_device();
        let model_state = make_model_state(&device);

        let mut checkpointer = AsyncCheckpointer::new();

        // First save
        checkpointer
            .save(dir1.path(), &model_state, None, &make_training_state(1))
            .unwrap();

        // Second save — should wait for first to finish
        checkpointer
            .save(dir2.path(), &model_state, None, &make_training_state(2))
            .unwrap();

        checkpointer.wait().unwrap();

        // Both checkpoints should exist and be valid
        let (_, _, s1) = load_checkpoint::<CpuRuntime, _>(dir1.path(), &device).unwrap();
        let (_, _, s2) = load_checkpoint::<CpuRuntime, _>(dir2.path(), &device).unwrap();
        assert_eq!(s1.step, 1);
        assert_eq!(s2.step, 2);
    }
}
