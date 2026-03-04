use super::*;
use crate::trainer::checkpoint::load_checkpoint;
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
            world_size: 2,
            rank: 1,
            owned_params: vec!["head.weight".to_string()],
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
            world_size: 2,
            rank: 1,
            owned_params: vec!["head.weight".to_string()],
            strategy: ShardingStrategy::ZeroPartitioned { stage: 3 },
            split_dims: HashMap::new(),
        },
    )
    .unwrap();

    // Update sharding_meta.json with rank 1's info
    let meta_json = std::fs::read_to_string(dir.path().join("sharding_meta.json")).unwrap();
    let mut meta: ShardingMeta = serde_json::from_str(&meta_json).unwrap();
    meta.shards[1].owned_params = vec!["head.weight".to_string()];
    meta.shards[1].split_dims = HashMap::new();
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
            split_dims: HashMap::new(),
        },
    )
    .unwrap();

    // Try loading as rank 3 (out of range for world_size=2)
    let err = load_distributed_checkpoint::<CpuRuntime, _>(dir.path(), 3, &device).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("out of range"), "unexpected error: {msg}");
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
        ),
    );
    model_r0.insert(
        "norm.weight".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device),
    );

    // Rank 1: second half of wq weight [2, 4]
    let mut model_r1 = HashMap::new();
    model_r1.insert(
        "attn.wq".to_string(),
        Tensor::<CpuRuntime>::from_slice(
            &[9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[2, 4],
            &device,
        ),
    );
    model_r1.insert(
        "norm.weight".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device),
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
            world_size: 2,
            rank: 0,
            owned_params: vec!["attn.wq".to_string(), "norm.weight".to_string()],
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
            world_size: 2,
            rank: 1,
            owned_params: vec!["attn.wq".to_string(), "norm.weight".to_string()],
            strategy: ShardingStrategy::TensorParallel,
            split_dims: split_dims.clone(),
        },
    )
    .unwrap();

    // Update sharding_meta.json with rank 1's info
    let meta_json = std::fs::read_to_string(dir.path().join("sharding_meta.json")).unwrap();
    let mut meta: ShardingMeta = serde_json::from_str(&meta_json).unwrap();
    meta.shards[1].owned_params = vec!["attn.wq".to_string(), "norm.weight".to_string()];
    meta.shards[1].split_dims = split_dims;
    std::fs::write(
        dir.path().join("sharding_meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
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
