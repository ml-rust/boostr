//! Load a per-rank distributed checkpoint shard.

use crate::error::{Error, Result};
use crate::trainer::checkpoint::{CheckpointData, load_checkpoint};
use crate::trainer::distributed_checkpoint::types::ShardingMeta;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::path::Path;

/// Load a distributed checkpoint for a specific rank.
///
/// Expects the checkpoint to have been saved with the same world_size.
/// If the topology differs, returns an error suggesting `consolidate_checkpoint` first.
pub fn load_distributed_checkpoint<R: Runtime<DType = DType>, P: AsRef<Path>>(
    dir: P,
    rank: usize,
    device: &R::Device,
) -> Result<CheckpointData<R>> {
    let dir = dir.as_ref();

    // Read sharding metadata to validate topology
    let meta_path = dir.join("sharding_meta.json");
    if !meta_path.exists() {
        return Err(Error::TrainingError {
            reason: format!(
                "sharding_meta.json not found in {}: not a distributed checkpoint",
                dir.display()
            ),
        });
    }

    let meta_json = std::fs::read_to_string(&meta_path).map_err(|e| Error::TrainingError {
        reason: format!("failed to read sharding meta: {e}"),
    })?;
    let meta: ShardingMeta =
        serde_json::from_str(&meta_json).map_err(|e| Error::TrainingError {
            reason: format!("failed to parse sharding meta: {e}"),
        })?;

    if rank >= meta.world_size {
        return Err(Error::TrainingError {
            reason: format!(
                "rank {rank} out of range for checkpoint with world_size={}. \
                 Use consolidate_checkpoint to reshard.",
                meta.world_size
            ),
        });
    }

    // Load this rank's shard
    let rank_dir = dir.join(format!("rank_{rank}"));
    if !rank_dir.exists() {
        return Err(Error::TrainingError {
            reason: format!("rank_{rank} directory not found in {}", dir.display()),
        });
    }

    load_checkpoint::<R, _>(&rank_dir, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::distributed_checkpoint::save::save_distributed_checkpoint;
    use crate::trainer::distributed_checkpoint::types::{ShardingConfig, ShardingStrategy};
    use crate::trainer::test_helpers::*;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_distributed_save_and_load() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        // Rank 0 state
        let mut model_r0 = HashMap::new();
        model_r0.insert(
            "embed.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap(),
        );

        // Rank 1 state
        let mut model_r1 = HashMap::new();
        model_r1.insert(
            "head.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device).unwrap(),
        );

        let state = make_training_state(100);

        // Save both ranks
        save_distributed_checkpoint(
            dir.path(),
            0,
            2,
            &model_r0,
            None,
            &state,
            ShardingConfig {
                strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
                split_dims: HashMap::new(),
            },
        )
        .unwrap();

        save_distributed_checkpoint(
            dir.path(),
            1,
            2,
            &model_r1,
            None,
            &state,
            ShardingConfig {
                strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
                split_dims: HashMap::new(),
            },
        )
        .unwrap();

        // Verify directory structure
        assert!(dir.path().join("rank_0/model.safetensors").exists());
        assert!(dir.path().join("rank_1/model.safetensors").exists());
        assert!(dir.path().join("sharding_meta.json").exists());

        // Load each rank
        let (loaded_r0, _, _) =
            load_distributed_checkpoint::<CpuRuntime, _>(dir.path(), 0, &device).unwrap();
        assert!(loaded_r0.contains_key("embed.weight"));

        let (loaded_r1, _, _) =
            load_distributed_checkpoint::<CpuRuntime, _>(dir.path(), 1, &device).unwrap();
        assert!(loaded_r1.contains_key("head.weight"));
    }

    #[test]
    fn test_distributed_topology_mismatch() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        let mut model = HashMap::new();
        model.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
        );
        let state = make_training_state(1);

        save_distributed_checkpoint(
            dir.path(),
            0,
            2,
            &model,
            None,
            &state,
            ShardingConfig {
                strategy: ShardingStrategy::Replicated,
                split_dims: HashMap::new(),
            },
        )
        .unwrap();

        // Try loading as rank 3 (out of range for world_size=2)
        let err = load_distributed_checkpoint::<CpuRuntime, _>(dir.path(), 3, &device).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("out of range"), "unexpected error: {msg}");
    }
}
