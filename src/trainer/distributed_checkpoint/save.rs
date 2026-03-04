//! Save a per-rank distributed checkpoint shard.

use crate::error::{Error, Result};
use crate::trainer::checkpoint::{CHECKPOINT_VERSION, TrainingState, save_checkpoint};
use crate::trainer::distributed_checkpoint::types::{ShardingConfig, ShardingMeta};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

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
                            split_dims: HashMap::new(),
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
