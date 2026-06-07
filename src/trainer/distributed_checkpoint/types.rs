//! Sharding configuration types for distributed checkpoints.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The sharding scheme used to write a distributed checkpoint.
///
/// This describes *how* the checkpoint is sharded, which is identical across
/// ranks: every rank passes the same `strategy` (and, for `TensorParallel`, the
/// same `split_dims` map) to `save_distributed_checkpoint`. It is not a per-rank
/// record — what each rank actually owns is implicit in the tensors it saves to
/// its own `rank_{n}/` directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
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

/// Metadata written once (by rank 0) describing a sharded checkpoint.
///
/// The sharding scheme is global, so this stores it once rather than as a
/// per-rank list. Each rank's actual data lives in its own `rank_{n}/`
/// directory; consolidation reads those tensors directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingMeta {
    /// Checkpoint format version.
    pub version: u32,
    /// Total number of ranks.
    pub world_size: usize,
    /// Sharding strategy used across all ranks.
    pub strategy: ShardingStrategy,
    /// For TensorParallel: map from param name to the dimension it was split
    /// along. Empty for other strategies.
    #[serde(default)]
    pub split_dims: HashMap<String, usize>,
}
