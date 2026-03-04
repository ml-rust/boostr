//! Sharding configuration types for distributed checkpoints.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// For TensorParallel: map from param name to the dimension it was split along.
    /// e.g. {"attention.wq": 0, "attention.wk": 0, "mlp.w1": 1}
    #[serde(default)]
    pub split_dims: HashMap<String, usize>,
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
