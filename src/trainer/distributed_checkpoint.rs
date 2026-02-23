//! Distributed checkpointing — per-rank shards with resharding metadata.

use crate::error::{Error, Result};
use crate::trainer::checkpoint::{
    CHECKPOINT_VERSION, CheckpointData, TrainingState, load_checkpoint, save_checkpoint,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Describes how a checkpoint was sharded across ranks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Total number of ranks that participated in saving.
    pub world_size: usize,
    /// This rank's index.
    pub rank: usize,
    /// Parameter names this rank owns.
    pub owned_params: Vec<String>,
    /// Sharding strategy used.
    pub strategy: ShardingStrategy,
}

/// Sharding strategy for distributed checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Full replication — every rank has all params (no sharding).
    Replicated,
    /// ZeRO-style optimizer state partitioning.
    ZeroPartitioned { stage: u8 },
    /// Tensor parallel — params are sliced along a dimension.
    TensorParallel,
}

/// Metadata written by rank 0 describing the full sharded checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingMeta {
    /// Checkpoint format version.
    pub version: u32,
    /// Total number of ranks.
    pub world_size: usize,
    /// Per-rank sharding configs.
    pub shards: Vec<ShardingConfig>,
}

/// Save a distributed checkpoint. Each rank saves its own shard.
///
/// Layout:
/// - `{dir}/rank_{rank}/model.safetensors`
/// - `{dir}/rank_{rank}/optimizer.safetensors`
/// - `{dir}/rank_{rank}/training_state.json`
/// - `{dir}/sharding_meta.json` (written by rank 0 — caller must coordinate)
///
/// Callers are responsible for barrier synchronization across ranks after this returns.
#[allow(clippy::too_many_arguments)]
pub fn save_distributed_checkpoint<P: AsRef<Path>>(
    dir: P,
    rank: usize,
    world_size: usize,
    model_state: &HashMap<String, Tensor<CpuRuntime>>,
    optimizer_state: Option<&HashMap<String, Tensor<CpuRuntime>>>,
    training_state: &TrainingState,
    sharding: ShardingConfig,
) -> Result<()> {
    let dir = dir.as_ref();
    let rank_dir = dir.join(format!("rank_{rank}"));

    // Save this rank's shard using the standard save_checkpoint
    save_checkpoint(&rank_dir, model_state, optimizer_state, training_state)?;

    // Rank 0 writes the sharding metadata
    if rank == 0 {
        let meta = ShardingMeta {
            version: CHECKPOINT_VERSION,
            world_size,
            shards: (0..world_size)
                .map(|r| {
                    if r == rank {
                        sharding.clone()
                    } else {
                        // Other ranks' configs will be filled in by those ranks
                        // or by a coordinated gather. For now, rank 0 writes a
                        // placeholder that can be updated.
                        ShardingConfig {
                            world_size,
                            rank: r,
                            owned_params: Vec::new(),
                            strategy: sharding.strategy.clone(),
                        }
                    }
                })
                .collect(),
        };

        let json = serde_json::to_string_pretty(&meta).map_err(|e| Error::TrainingError {
            reason: format!("failed to serialize sharding meta: {e}"),
        })?;
        std::fs::create_dir_all(dir).map_err(|e| Error::TrainingError {
            reason: format!("failed to create checkpoint dir: {e}"),
        })?;
        std::fs::write(dir.join("sharding_meta.json"), json).map_err(|e| Error::TrainingError {
            reason: format!("failed to write sharding meta: {e}"),
        })?;
    }

    Ok(())
}

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

/// Consolidate a sharded checkpoint into a single (non-distributed) checkpoint.
///
/// Reads all rank shards and merges them into one `model.safetensors` + `optimizer.safetensors`.
/// Useful for: (1) resharding to different world_size, (2) exporting for inference,
/// (3) migrating between distributed strategies.
///
/// For `Replicated` strategy, uses rank 0's data.
/// For `ZeroPartitioned`, merges each rank's owned params.
/// For `TensorParallel`, concatenation along the split dimension would require
/// knowing the split dim per tensor — currently returns an error (future work).
pub fn consolidate_checkpoint<P: AsRef<Path>>(sharded_dir: P, output_dir: P) -> Result<()> {
    let sharded_dir = sharded_dir.as_ref();
    let output_dir = output_dir.as_ref();

    // Read sharding metadata
    let meta_path = sharded_dir.join("sharding_meta.json");
    let meta_json = std::fs::read_to_string(&meta_path).map_err(|e| Error::TrainingError {
        reason: format!("failed to read sharding meta: {e}"),
    })?;
    let meta: ShardingMeta =
        serde_json::from_str(&meta_json).map_err(|e| Error::TrainingError {
            reason: format!("failed to parse sharding meta: {e}"),
        })?;

    let device = CpuDevice::new();
    let mut merged_model: HashMap<String, Tensor<CpuRuntime>> = HashMap::new();
    let mut merged_opt: HashMap<String, Tensor<CpuRuntime>> = HashMap::new();
    let mut training_state: Option<TrainingState> = None;

    for rank in 0..meta.world_size {
        let rank_dir = sharded_dir.join(format!("rank_{rank}"));
        let (model, opt, state) = load_checkpoint::<CpuRuntime, _>(&rank_dir, &device)?;

        if training_state.is_none() {
            training_state = Some(state);
        }

        let strategy = &meta.shards[rank].strategy;
        match strategy {
            ShardingStrategy::Replicated => {
                // Use rank 0's data only
                if rank == 0 {
                    merged_model = model;
                    if let Some(opt) = opt {
                        merged_opt = opt;
                    }
                }
            }
            ShardingStrategy::ZeroPartitioned { .. } => {
                // Each rank owns a disjoint subset of params
                merged_model.extend(model);
                if let Some(opt) = opt {
                    merged_opt.extend(opt);
                }
            }
            ShardingStrategy::TensorParallel => {
                return Err(Error::TrainingError {
                    reason: "consolidation of TensorParallel checkpoints requires \
                             per-tensor split dimension metadata (not yet implemented)"
                        .to_string(),
                });
            }
        }
    }

    let state = training_state.ok_or_else(|| Error::TrainingError {
        reason: "no ranks found in sharded checkpoint".to_string(),
    })?;

    save_checkpoint(
        output_dir,
        &merged_model,
        if merged_opt.is_empty() {
            None
        } else {
            Some(&merged_opt)
        },
        &state,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::test_helpers::*;
    use numr::runtime::cpu::CpuRuntime;
    use tempfile::TempDir;

    #[test]
    fn test_distributed_save_and_load() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        // Rank 0 state
        let mut model_r0 = HashMap::new();
        model_r0.insert(
            "embed.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
        );

        // Rank 1 state
        let mut model_r1 = HashMap::new();
        model_r1.insert(
            "head.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device),
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
                world_size: 2,
                rank: 0,
                owned_params: vec!["embed.weight".to_string()],
                strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
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
                world_size: 2,
                rank: 1,
                owned_params: vec!["head.weight".to_string()],
                strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
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
    fn test_consolidate_zero_partitioned() {
        let dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let device = make_device();

        // Rank 0 owns embed.weight
        let mut model_r0 = HashMap::new();
        model_r0.insert(
            "embed.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device),
        );

        // Rank 1 owns head.weight
        let mut model_r1 = HashMap::new();
        model_r1.insert(
            "head.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device),
        );

        let state = make_training_state(200);

        save_distributed_checkpoint(
            dir.path(),
            0,
            2,
            &model_r0,
            None,
            &state,
            ShardingConfig {
                world_size: 2,
                rank: 0,
                owned_params: vec!["embed.weight".to_string()],
                strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
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
                world_size: 2,
                rank: 1,
                owned_params: vec!["head.weight".to_string()],
                strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
            },
        )
        .unwrap();

        // Update sharding_meta.json with rank 1's info
        let meta_json = std::fs::read_to_string(dir.path().join("sharding_meta.json")).unwrap();
        let mut meta: ShardingMeta = serde_json::from_str(&meta_json).unwrap();
        meta.shards[1].owned_params = vec!["head.weight".to_string()];
        std::fs::write(
            dir.path().join("sharding_meta.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();

        // Consolidate
        consolidate_checkpoint(dir.path(), output_dir.path()).unwrap();

        // Verify merged checkpoint has both params
        let (merged, _, merged_state) =
            load_checkpoint::<CpuRuntime, _>(output_dir.path(), &device).unwrap();
        assert_eq!(merged.len(), 2);
        assert!(merged.contains_key("embed.weight"));
        assert!(merged.contains_key("head.weight"));
        assert_eq!(merged_state.step, 200);

        let embed: Vec<f32> = merged["embed.weight"].to_vec();
        assert!((embed[0] - 1.0).abs() < 1e-6);
        let head: Vec<f32> = merged["head.weight"].to_vec();
        assert!((head[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_distributed_topology_mismatch() {
        let dir = TempDir::new().unwrap();
        let device = make_device();

        let mut model = HashMap::new();
        model.insert(
            "w".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
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
                world_size: 2,
                rank: 0,
                owned_params: vec!["w".to_string()],
                strategy: ShardingStrategy::Replicated,
            },
        )
        .unwrap();

        // Try loading as rank 3 (out of range for world_size=2)
        let err = load_distributed_checkpoint::<CpuRuntime, _>(dir.path(), 3, &device).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("out of range"), "unexpected error: {msg}");
    }
}
