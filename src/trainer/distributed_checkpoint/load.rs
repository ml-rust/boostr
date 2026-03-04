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
