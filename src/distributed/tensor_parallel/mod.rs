//! Tensor parallelism (Megatron-LM style)
//!
//! Column-parallel and row-parallel linear layers that split model weights
//! across devices. Follows the Megatron-LM approach where column-parallel
//! splits output features and row-parallel splits input features.

mod column;
mod row;
mod shard;

pub use column::ColumnParallelLinear;
pub use row::RowParallelLinear;
pub use shard::{gather_from_ranks, scatter_to_rank};
