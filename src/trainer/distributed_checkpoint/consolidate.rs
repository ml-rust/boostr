//! Consolidate sharded distributed checkpoints into a single checkpoint.

use crate::error::{Error, Result};
use crate::trainer::checkpoint::{TrainingState, load_checkpoint, save_checkpoint};
use crate::trainer::distributed_checkpoint::types::{ShardingMeta, ShardingStrategy};
use numr::ops::ShapeOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Consolidate a sharded checkpoint into a single (non-distributed) checkpoint.
///
/// Reads all rank shards and merges them into one `model.safetensors` + `optimizer.safetensors`.
/// Useful for: (1) resharding to different world_size, (2) exporting for inference,
/// (3) migrating between distributed strategies.
///
/// For `Replicated` strategy, uses rank 0's data.
/// For `ZeroPartitioned`, merges each rank's owned params.
/// For `TensorParallel`, concatenates tensors along their split dimensions using metadata.
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

    // For TensorParallel, we need to collect all ranks before merging
    let strategy = &meta.shards[0].strategy;
    let is_tensor_parallel = matches!(strategy, ShardingStrategy::TensorParallel);

    if is_tensor_parallel {
        consolidate_tensor_parallel(
            &meta,
            sharded_dir,
            &device,
            &mut merged_model,
            &mut merged_opt,
            &mut training_state,
        )?;
    } else {
        // Replicated and ZeroPartitioned can be processed rank-by-rank
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
                    // Should not reach here (handled above)
                    unreachable!()
                }
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

/// Helper function to consolidate TensorParallel checkpoints by concatenating along split dims.
fn consolidate_tensor_parallel(
    meta: &ShardingMeta,
    sharded_dir: &Path,
    device: &CpuDevice,
    merged_model: &mut HashMap<String, Tensor<CpuRuntime>>,
    merged_opt: &mut HashMap<String, Tensor<CpuRuntime>>,
    training_state: &mut Option<TrainingState>,
) -> Result<()> {
    // Get split_dims from rank 0's config
    let split_dims = &meta.shards[0].split_dims;
    if split_dims.is_empty() {
        return Err(Error::TrainingError {
            reason: "TensorParallel consolidation requires split_dims metadata \
                     in ShardingConfig (missing for rank 0)"
                .to_string(),
        });
    }

    // Load all rank data
    let mut all_models = Vec::new();
    let mut all_opts = Vec::new();

    for rank in 0..meta.world_size {
        let rank_dir = sharded_dir.join(format!("rank_{rank}"));
        let (model, opt, state) = load_checkpoint::<CpuRuntime, _>(&rank_dir, device)?;

        if training_state.is_none() {
            *training_state = Some(state);
        }

        all_models.push(model);
        all_opts.push(opt);
    }

    // Get client for concatenation
    let client = CpuClient::new(device.clone());

    // Merge model: concatenate along split dimension for each param
    for (param_name, &split_dim) in split_dims {
        let shards: Vec<&Tensor<CpuRuntime>> = (0..meta.world_size)
            .filter_map(|r| all_models[r].get(param_name))
            .collect();

        if shards.len() == meta.world_size {
            let merged =
                client
                    .cat(&shards, split_dim as isize)
                    .map_err(|e| Error::TrainingError {
                        reason: format!(
                            "failed to concatenate param '{}' along dim {}: {}",
                            param_name, split_dim, e
                        ),
                    })?;
            merged_model.insert(param_name.clone(), merged);
        }
    }

    // Copy non-split params from rank 0 (replicated params like norms, biases)
    if let Some(rank0_model) = all_models.first() {
        for (name, tensor) in rank0_model {
            if !split_dims.contains_key(name) {
                merged_model.insert(name.clone(), tensor.clone());
            }
        }
    }

    // Handle optimizer states similarly
    if let Some(Some(opt0)) = all_opts.first() {
        for (name, tensor) in opt0 {
            if split_dims.contains_key(name) {
                // Collect this optimizer state from all ranks
                let shards: Vec<&Tensor<CpuRuntime>> = (0..meta.world_size)
                    .filter_map(|r| all_opts[r].as_ref().and_then(|o| o.get(name)))
                    .collect();

                if shards.len() == meta.world_size {
                    let split_dim = split_dims[name];
                    if let Ok(merged) = client.cat(&shards, split_dim as isize) {
                        merged_opt.insert(name.clone(), merged);
                    }
                }
            } else {
                // Non-split optimizer state from rank 0
                merged_opt.insert(name.clone(), tensor.clone());
            }
        }
    }

    Ok(())
}
