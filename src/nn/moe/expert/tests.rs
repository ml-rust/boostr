use super::*;
use crate::nn::lora::LoraLinear;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

const HIDDEN: usize = 4;
const INTER: usize = 6;
const RANK: usize = 2;
const ALPHA: f32 = 4.0;
const TOKENS: usize = 2;

/// Asymmetric, non-zero values so an accidental zero cannot make a test pass.
fn vals(count: usize, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 * 0.13 + offset).sin() * 0.5)
        .collect()
}

fn tensor(data: &[f32], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(data, shape, device)
}

/// The three plain SwiGLU projections, built from fixed data.
fn projections(
    device: &CpuDevice,
    trainable: bool,
) -> (Linear<CpuRuntime>, Linear<CpuRuntime>, Linear<CpuRuntime>) {
    let gate = tensor(&vals(INTER * HIDDEN, 0.0), &[INTER, HIDDEN], device);
    let up = tensor(&vals(INTER * HIDDEN, 1.0), &[INTER, HIDDEN], device);
    let down = tensor(&vals(HIDDEN * INTER, 2.0), &[HIDDEN, INTER], device);
    (
        Linear::new(gate, None, trainable),
        Linear::new(up, None, trainable),
        Linear::new(down, None, trainable),
    )
}

fn input(device: &CpuDevice) -> Var<CpuRuntime> {
    Var::new(
        tensor(&vals(TOKENS * HIDDEN, 3.0), &[TOKENS, HIDDEN], device),
        false,
    )
}

/// Wrap a projection in a LoRA adapter with a NON-ZERO `lora_b`.
///
/// `LoraLinear::new` zero-initializes `lora_b`, which makes the adapted forward
/// identical to the plain one — a test built on it proves nothing.
fn adapt(base: Linear<CpuRuntime>, device: &CpuDevice, offset: f32) -> MaybeLoraLinear<CpuRuntime> {
    let in_features = base.weight().tensor().shape()[1];
    let out_features = base.weight().tensor().shape()[0];
    let a = tensor(
        &vals(RANK * in_features, offset),
        &[RANK, in_features],
        device,
    );
    let b = tensor(
        &vals(out_features * RANK, offset + 0.7),
        &[out_features, RANK],
        device,
    );
    LoraLinear::from_weights(base, a, b, ALPHA, true).into()
}

#[test]
fn test_expert_forward_shape() {
    let (client, device) = cpu_setup();
    let gate_w = tensor(&[0.1f32; INTER * HIDDEN], &[INTER, HIDDEN], &device);
    let up_w = tensor(&[0.1f32; INTER * HIDDEN], &[INTER, HIDDEN], &device);
    let down_w = tensor(&[0.1f32; HIDDEN * INTER], &[HIDDEN, INTER], &device);

    let expert = Expert::from_tensors(gate_w, up_w, down_w, false);

    let x = Var::new(
        tensor(&[1.0f32; TOKENS * HIDDEN], &[TOKENS, HIDDEN], &device),
        false,
    );
    let out = expert.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[TOKENS, HIDDEN]);
}

/// `Expert::new` with plain `Linear`s must behave exactly as before the
/// projections became `MaybeLoraLinear` — same SwiGLU composition, bit for bit.
#[test]
fn test_plain_expert_matches_direct_linear_path() {
    let (client, device) = cpu_setup();
    let (gate, up, down) = projections(&device, false);
    let x = input(&device);

    // Reference: the pre-change body, computed on standalone `Linear`s.
    let (ref_gate, ref_up, ref_down) = projections(&device, false);
    let g = ref_gate.forward(&client, &x).unwrap();
    let u = ref_up.forward(&client, &x).unwrap();
    let g_silu = var_silu(&g, &client).unwrap();
    let hidden = var_mul(&g_silu, &u, &client).unwrap();
    let expected: Vec<f32> = ref_down
        .forward(&client, &hidden)
        .unwrap()
        .tensor()
        .to_vec();

    let expert = Expert::new(gate, up, down);
    let actual: Vec<f32> = expert.forward(&client, &x).unwrap().tensor().to_vec();

    assert_eq!(actual, expected);
}

/// An adapted expert exposes its base weights AND both adapter factors.
#[test]
fn test_adapted_expert_parameters_include_adapters_and_bases() {
    let (_client, device) = cpu_setup();
    let (gate, up, down) = projections(&device, true);
    let expert = Expert::new_adapted(
        adapt(gate, &device, 0.4),
        adapt(up, &device, 0.9),
        down.into(),
    );

    let params = Module::parameters_with_ids(&expert);
    // 3 base weights (no biases) + 2 adapters x 2 factors.
    assert_eq!(params.len(), 7);

    for proj in [expert.gate_proj(), expert.up_proj()] {
        let (a, b) = proj.adapters().expect("projection is adapted");
        assert!(params.iter().any(|(id, _)| *id == a.id()));
        assert!(params.iter().any(|(id, _)| *id == b.id()));
        assert!(params.iter().any(|(id, _)| *id == proj.weight().id()));
    }
    assert!(expert.down_proj().adapters().is_none());
    assert!(
        params
            .iter()
            .any(|(id, _)| *id == expert.down_proj().weight().id())
    );
}

/// With frozen bases, only the adapter factors are trainable.
#[test]
fn test_frozen_bases_leave_only_adapters_trainable() {
    let (_client, device) = cpu_setup();
    let (gate, up, down) = projections(&device, false);
    let expert = Expert::new_adapted(
        adapt(gate, &device, 0.4),
        adapt(up, &device, 0.9),
        down.into(),
    );

    let trainable = Module::trainable_parameters(&expert);
    assert_eq!(trainable.len(), 4);

    let mut expected_ids = Vec::new();
    for proj in [expert.gate_proj(), expert.up_proj()] {
        let (a, b) = proj.adapters().expect("projection is adapted");
        expected_ids.push(a.id());
        expected_ids.push(b.id());
    }
    for (id, var) in &trainable {
        assert!(expected_ids.contains(id));
        assert!(var.requires_grad());
    }
}

/// A non-zero `lora_b` must change the output; otherwise the adapter is inert.
#[test]
fn test_adapted_forward_differs_from_plain() {
    let (client, device) = cpu_setup();
    let x = input(&device);

    let (gate, up, down) = projections(&device, false);
    let plain: Vec<f32> = Expert::new(gate, up, down)
        .forward(&client, &x)
        .unwrap()
        .tensor()
        .to_vec();

    let (gate, up, down) = projections(&device, false);
    let adapted: Vec<f32> = Expert::new_adapted(
        adapt(gate, &device, 0.4),
        adapt(up, &device, 0.9),
        adapt(down, &device, 1.5),
    )
    .forward(&client, &x)
    .unwrap()
    .tensor()
    .to_vec();

    assert_eq!(plain.len(), adapted.len());
    let max_diff = plain
        .iter()
        .zip(&adapted)
        .map(|(p, a)| (p - a).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "adapter had no effect (max diff {max_diff})"
    );
}

/// Merging folds the adapters into the base weights without changing the output.
#[test]
fn test_merge_adapters_preserves_forward() {
    let (client, device) = cpu_setup();
    let x = input(&device);
    let (gate, up, down) = projections(&device, false);
    let expert = Expert::new_adapted(
        adapt(gate, &device, 0.4),
        adapt(up, &device, 0.9),
        adapt(down, &device, 1.5),
    );

    let adapted: Vec<f32> = expert.forward(&client, &x).unwrap().tensor().to_vec();

    let merged = expert.merge_adapters(&client).unwrap();
    assert!(merged.gate_proj().adapters().is_none());
    assert!(merged.up_proj().adapters().is_none());
    assert!(merged.down_proj().adapters().is_none());

    let merged_out: Vec<f32> = merged.forward(&client, &x).unwrap().tensor().to_vec();
    assert_eq!(adapted.len(), merged_out.len());
    for (a, m) in adapted.iter().zip(&merged_out) {
        assert!((a - m).abs() < 1e-6, "merged output diverged: {a} vs {m}");
    }
}
