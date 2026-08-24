use super::*;
use crate::quant::QuantFormat;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

fn device() -> CpuDevice {
    CpuDevice::new()
}

fn client() -> numr::runtime::cpu::CpuClient {
    let d = device();
    CpuRuntime::default_client(&d)
}

#[test]
fn test_varbuilder_prefix() {
    let d = device();
    let mut map = VarMap::<CpuRuntime>::new();
    map.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        Tensor::from_slice(&[1.0f32], &[1], &d).unwrap(),
    );

    let mut vb = VarBuilder::new(&mut map, &d);
    let mut vb = vb.pp("model");
    let mut vb = vb.pp("layers");
    let mut vb = vb.pp("0");
    let vb = vb.pp("self_attn");
    let t = vb.get_tensor("q_proj.weight").unwrap();
    assert_eq!(t.shape(), &[1]);
}

#[test]
fn test_varbuilder_get_with_shape() {
    let d = device();
    let mut map = VarMap::<CpuRuntime>::new();
    map.insert(
        "w".into(),
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &d).unwrap(),
    );

    let vb = VarBuilder::new(&mut map, &d);
    assert!(vb.get_with_shape("w", &[2, 2]).is_ok());
    assert!(vb.get_with_shape("w", &[4]).is_err());
}

#[test]
fn test_varbuilder_take_tensor() {
    let d = device();
    let mut map = VarMap::<CpuRuntime>::new();
    map.insert(
        "layer.weight".into(),
        Tensor::from_slice(&[1.0f32, 2.0], &[2], &d).unwrap(),
    );

    let mut vb = VarBuilder::new(&mut map, &d);
    let mut vb = vb.pp("layer");
    let t = vb.take_tensor("weight").unwrap();
    assert_eq!(t.shape(), &[2]);
    // Second take should fail — already removed
    assert!(vb.take_tensor("weight").is_err());
}

#[test]
fn test_varbuilder_take_tensor_shard() {
    let d = device();
    let mut map = VarMap::<CpuRuntime>::new();
    // [4, 6] weight
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    map.insert(
        "weight".into(),
        Tensor::from_slice(&data, &[4, 6], &d).unwrap(),
    );

    let vb = VarBuilder::new(&mut map, &d);

    // Column-parallel shard (dim=0, rank=0, world_size=2) → [2, 6]
    // Re-insert since take removes it
    let data2: Vec<f32> = (0..24).map(|i| i as f32).collect();
    drop(vb);
    map.insert(
        "weight".into(),
        Tensor::from_slice(&data2, &[4, 6], &d).unwrap(),
    );
    let mut vb = VarBuilder::new(&mut map, &d);
    let shard = vb.take_tensor_shard("weight", 0, 0, 2).unwrap();
    assert_eq!(shard.shape(), &[2, 6]);

    // Row-parallel shard (dim=1, rank=1, world_size=2) → [4, 3]
    let data3: Vec<f32> = (0..24).map(|i| i as f32).collect();
    drop(vb);
    map.insert(
        "weight".into(),
        Tensor::from_slice(&data3, &[4, 6], &d).unwrap(),
    );
    let mut vb = VarBuilder::new(&mut map, &d);
    let shard = vb.take_tensor_shard("weight", 1, 1, 2).unwrap();
    assert_eq!(shard.shape(), &[4, 3]);
}

#[test]
fn test_varbuilder_take_tensor_shard_not_divisible() {
    let d = device();
    let mut map = VarMap::<CpuRuntime>::new();
    map.insert(
        "weight".into(),
        Tensor::from_slice(&[1.0f32; 15], &[3, 5], &d).unwrap(),
    );
    let mut vb = VarBuilder::new(&mut map, &d);
    // 3 not divisible by 2
    assert!(vb.take_tensor_shard("weight", 0, 0, 2).is_err());
}

#[test]
fn test_varbuilder_quant_prefix() {
    let d = device();
    let mut map = VarMap::<CpuRuntime>::new();
    let data = vec![0u8; 18];
    let qt = QuantTensor::from_bytes(&data, QuantFormat::Q4_0, &[32], &d).unwrap();
    map.insert_quant("layers.0.weight".into(), qt);

    let mut vb = VarBuilder::new(&mut map, &d);
    let mut vb = vb.pp("layers");
    let vb = vb.pp("0");
    let qt = vb.get_quant_tensor("weight").unwrap();
    assert_eq!(qt.shape(), &[32]);
}

// ===== Seeded-path tests =====

/// Two builders seeded identically must initialize the same (never-loaded)
/// parameter to bit-identical values.
#[test]
fn test_varbuilder_with_seed_reproducible() {
    let d = device();
    let c = client();

    let mut map_a = VarMap::<CpuRuntime>::new();
    let mut vb_a = VarBuilder::new(&mut map_a, &d).with_seed(123);
    let t_a = vb_a
        .take_or_init_tensor(
            "weight",
            &[128, 64],
            DType::F32,
            crate::nn::Init::Kaiming,
            &c,
        )
        .unwrap();

    let mut map_b = VarMap::<CpuRuntime>::new();
    let mut vb_b = VarBuilder::new(&mut map_b, &d).with_seed(123);
    let t_b = vb_b
        .take_or_init_tensor(
            "weight",
            &[128, 64],
            DType::F32,
            crate::nn::Init::Kaiming,
            &c,
        )
        .unwrap();

    assert_eq!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
}

/// The seed must survive `pp()` into a nested prefix: a child builder derived
/// from a seeded parent must ALSO produce reproducible init, not silently
/// fall back to unseeded randomness.
#[test]
fn test_varbuilder_seed_survives_push_prefix() {
    let d = device();
    let c = client();

    let mut map_a = VarMap::<CpuRuntime>::new();
    let mut root_a = VarBuilder::new(&mut map_a, &d).with_seed(456);
    let mut vb_a = root_a.pp("layers");
    let mut vb_a = vb_a.pp("0");
    let t_a = vb_a
        .take_or_init_tensor(
            "weight",
            &[32, 32],
            DType::F32,
            crate::nn::Init::PyTorchLinear,
            &c,
        )
        .unwrap();

    let mut map_b = VarMap::<CpuRuntime>::new();
    let mut root_b = VarBuilder::new(&mut map_b, &d).with_seed(456);
    let mut vb_b = root_b.pp("layers");
    let mut vb_b = vb_b.pp("0");
    let t_b = vb_b
        .take_or_init_tensor(
            "weight",
            &[32, 32],
            DType::F32,
            crate::nn::Init::PyTorchLinear,
            &c,
        )
        .unwrap();

    assert_eq!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
}

/// Per-tensor seeds are derived from the NAME, not from call order: two
/// differently-named tensors under the same base seed must get different
/// values. If seeding instead used a shared incrementing counter, this
/// would fail (both would land on the same counter value from a fresh
/// builder), which is exactly the order-dependence bug name-derivation
/// avoids.
#[test]
fn test_varbuilder_seed_is_name_derived_not_order_derived() {
    let d = device();
    let c = client();

    let mut map = VarMap::<CpuRuntime>::new();
    let mut vb = VarBuilder::new(&mut map, &d).with_seed(789);
    let t_a = vb
        .take_or_init_tensor(
            "weight_a",
            &[4096],
            DType::F32,
            crate::nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
            &c,
        )
        .unwrap();
    let t_b = vb
        .take_or_init_tensor(
            "weight_b",
            &[4096],
            DType::F32,
            crate::nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
            &c,
        )
        .unwrap();
    assert_ne!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
}

/// A builder with no seed set behaves exactly like before `with_seed`
/// existed: repeated init calls are NOT reproducible.
#[test]
fn test_varbuilder_without_seed_stays_unseeded() {
    let d = device();
    let c = client();

    let mut map_a = VarMap::<CpuRuntime>::new();
    let mut vb_a = VarBuilder::new(&mut map_a, &d);
    let t_a = vb_a
        .take_or_init_tensor(
            "weight",
            &[4096],
            DType::F32,
            crate::nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
            &c,
        )
        .unwrap();

    let mut map_b = VarMap::<CpuRuntime>::new();
    let mut vb_b = VarBuilder::new(&mut map_b, &d);
    let t_b = vb_b
        .take_or_init_tensor(
            "weight",
            &[4096],
            DType::F32,
            crate::nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
            &c,
        )
        .unwrap();

    assert_ne!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
}
