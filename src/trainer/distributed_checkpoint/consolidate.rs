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
    let is_tensor_parallel = matches!(meta.strategy, ShardingStrategy::TensorParallel);

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

            match &meta.strategy {
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
    let split_dims = &meta.split_dims;
    if split_dims.is_empty() {
        return Err(Error::TrainingError {
            reason: "TensorParallel consolidation requires split_dims metadata".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::distributed_checkpoint::save::save_distributed_checkpoint;
    use crate::trainer::distributed_checkpoint::types::ShardingConfig;
    use crate::trainer::test_helpers::*;
    use tempfile::TempDir;

    #[test]
    fn test_consolidate_zero_partitioned() {
        let dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let device = make_device();

        // Rank 0 owns embed.weight
        let mut model_r0 = HashMap::new();
        model_r0.insert(
            "embed.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap(),
        );

        // Rank 1 owns head.weight
        let mut model_r1 = HashMap::new();
        model_r1.insert(
            "head.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap(),
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
    fn test_consolidate_tensor_parallel() {
        let dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let device = make_device();

        // Rank 0: first half of wq weight [2, 4]
        let mut model_r0 = HashMap::new();
        model_r0.insert(
            "attn.wq".to_string(),
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[2, 4],
                &device,
            )
            .unwrap(),
        );
        model_r0.insert(
            "norm.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap(),
        );

        // Rank 1: second half of wq weight [2, 4]
        let mut model_r1 = HashMap::new();
        model_r1.insert(
            "attn.wq".to_string(),
            Tensor::<CpuRuntime>::from_slice(
                &[9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                &[2, 4],
                &device,
            )
            .unwrap(),
        );
        model_r1.insert(
            "norm.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap(),
        );

        let state = make_training_state(300);

        let mut split_dims = HashMap::new();
        split_dims.insert("attn.wq".to_string(), 0usize);

        save_distributed_checkpoint(
            dir.path(),
            0,
            2,
            &model_r0,
            None,
            &state,
            ShardingConfig {
                strategy: ShardingStrategy::TensorParallel,
                split_dims: split_dims.clone(),
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
                strategy: ShardingStrategy::TensorParallel,
                split_dims: split_dims.clone(),
            },
        )
        .unwrap();

        // Consolidate
        consolidate_checkpoint(dir.path(), output_dir.path()).unwrap();

        let (merged, _, merged_state) =
            load_checkpoint::<CpuRuntime, _>(output_dir.path(), &device).unwrap();

        // attn.wq should be [4, 4] (concatenated along dim 0)
        let wq = &merged["attn.wq"];
        assert_eq!(wq.shape(), &[4, 4]);
        let wq_data: Vec<f32> = wq.to_vec();
        // First row from rank 0 should be [1, 2, 3, 4]
        assert!((wq_data[0] - 1.0).abs() < 1e-6);
        assert!((wq_data[1] - 2.0).abs() < 1e-6);
        // First row from rank 1 (at index 8) should be [9, 10, 11, 12]
        assert!((wq_data[8] - 9.0).abs() < 1e-6);
        assert!((wq_data[9] - 10.0).abs() < 1e-6);

        // norm.weight should be [2] (replicated, taken from rank 0)
        let norm = &merged["norm.weight"];
        assert_eq!(norm.shape(), &[2]);
        let norm_data: Vec<f32> = norm.to_vec();
        assert!((norm_data[0] - 1.0).abs() < 1e-6);

        // Training state should be preserved
        assert_eq!(merged_state.step, 300);
    }
}
