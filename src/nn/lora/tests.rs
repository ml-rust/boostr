use super::*;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn test_lora_linear_creation() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight: Tensor<CpuRuntime> = Tensor::zeros(&[64, 32], DType::F32, &device);
    let base = Linear::new(weight, None, false);
    let lora = LoraLinear::new(base, 8, 16.0, &device);
    assert_eq!(lora.rank(), 8);
    assert!((lora.scaling() - 2.0).abs() < 1e-6); // alpha/rank = 16/8 = 2
}

/// Gradients must reach the LoRA factors.
///
/// Regression: the scale-and-add tail was built with `Var::new(...)` on raw
/// tensors, producing a LEAF with no grad_fn. Backward then reached neither
/// `lora_a`/`lora_b` nor the base, so EVERY LoRA adapter silently never
/// trained — no error, no NaN, and the loss still falls because the rest of
/// the network learns.
#[test]
fn test_lora_forward_propagates_gradient_to_factors() {
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};

    let (client, device) = cpu_setup();
    let (in_features, out_features, rank) = (4usize, 3usize, 2usize);

    // Asymmetric weights so a genuine zero gradient cannot pass by accident.
    let base_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32) * 0.1 - 0.5)
        .collect();
    let base = Linear::new(
        Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device),
        None,
        false,
    );
    let lora = LoraLinear::new(base, rank, 16.0, &device);

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32) * 0.25 - 0.75)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device),
        false,
    );

    let out = lora.forward(&client, &x).expect("lora forward");
    let loss = var_sum(&out, &[0, 1], false, &client).expect("reduce");
    let grads = backward(&loss, &client).expect("backward");

    // lora_b is zero-initialised, so d(loss)/d(lora_a) is zero at step 0 by
    // construction; lora_b is the factor that must receive signal immediately.
    let b_grad = grads
        .get(lora.lora_b.id())
        .expect("lora_b must receive a gradient");
    let b_vals: Vec<f32> = b_grad.contiguous().expect("contig").to_vec();
    let magnitude: f32 = b_vals.iter().map(|v| v.abs()).sum();
    assert!(
        magnitude > 1e-8,
        "lora_b gradient is all zeros ({magnitude}) — the LoRA graph is severed"
    );

    // And lora_a must at least be reachable in the graph.
    assert!(
        grads.get(lora.lora_a.id()).is_some(),
        "lora_a must be reachable from the loss"
    );
}

#[test]
fn test_module_parameters_enumerates_base_and_adapters() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device);
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
    let base = Linear::new(weight, Some(bias), true);
    let lora = LoraLinear::new(base, 2, 4.0, &device);

    // base.weight + base.bias + lora_a + lora_b
    assert_eq!(lora.parameters().len(), 4);

    let named = lora.named_parameters();
    assert_eq!(named.len(), 4);
    assert!(named.iter().any(|(n, _)| n == "lora_a"));
    assert!(named.iter().any(|(n, _)| n == "lora_b"));
    assert!(named.iter().any(|(n, _)| n == "base.weight"));
    assert!(named.iter().any(|(n, _)| n == "base.bias"));
}

/// With a frozen base, only the adapter factors are trainable — this is
/// what lets LoRA train an adapter alone.
#[test]
fn test_trainable_parameters_excludes_frozen_base() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device);
    let base = Linear::new(weight, None, false); // frozen
    let lora = LoraLinear::new(base, 2, 4.0, &device);

    let trainable = lora.trainable_parameters();
    assert_eq!(trainable.len(), 2);
    assert_eq!(trainable[0].0, lora.lora_a.id());
    assert_eq!(trainable[1].0, lora.lora_b.id());
}

#[test]
fn test_from_weights_trainable_flag() {
    let device = <CpuRuntime as Runtime>::default_device();
    let make_base = || {
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device);
        Linear::new(weight, None, false)
    };
    let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device);

    let frozen = LoraLinear::from_weights(make_base(), a.clone(), b.clone(), 4.0, false);
    assert!(!frozen.lora_a.requires_grad());
    assert!(!frozen.lora_b.requires_grad());

    let trainable = LoraLinear::from_weights(make_base(), a, b, 4.0, true);
    assert!(trainable.lora_a.requires_grad());
    assert!(trainable.lora_b.requires_grad());
}

/// Merging into a plain `Linear` must reproduce the adapted forward pass
/// exactly, and must carry the base bias over unchanged. Uses a non-zero
/// `lora_b` (via `from_weights`) — with the default zero-init the
/// equivalence would hold trivially and prove nothing.
#[test]
fn test_merge_matches_forward_and_preserves_bias() {
    use crate::test_utils::cpu_setup;

    let (client, device) = cpu_setup();
    let (in_features, out_features, rank) = (3usize, 2usize, 2usize);

    let base_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32) * 0.1 - 0.3)
        .collect();
    let bias_v: Vec<f32> = vec![0.05, -0.05];
    let base = Linear::new(
        Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device),
        Some(Tensor::<CpuRuntime>::from_slice(
            &bias_v,
            &[out_features],
            &device,
        )),
        false,
    );

    let a_vals: Vec<f32> = (0..rank * in_features)
        .map(|i| (i as f32) * 0.05 - 0.1)
        .collect();
    // Deliberately non-zero, unlike LoraLinear::new's zero-init.
    let b_vals: Vec<f32> = (0..out_features * rank)
        .map(|i| (i as f32) * 0.07 + 0.02)
        .collect();
    let lora_a = Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device);
    let lora_b = Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device);
    let lora = LoraLinear::from_weights(base, lora_a, lora_b, 8.0, false);

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32) * 0.2 - 0.4)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device),
        false,
    );

    let lora_out = lora.forward(&client, &x).expect("lora forward");
    let merged = lora.merge_into_base(&client).expect("merge");
    let merged_out = merged.forward(&client, &x).expect("merged forward");

    let lora_vals: Vec<f32> = lora_out.tensor().contiguous().expect("contig").to_vec();
    let merged_vals: Vec<f32> = merged_out.tensor().contiguous().expect("contig").to_vec();
    assert_eq!(lora_vals.len(), merged_vals.len());
    for (l, m) in lora_vals.iter().zip(merged_vals.iter()) {
        assert!((l - m).abs() < 1e-5, "lora={l} merged={m}");
    }

    let merged_bias: Vec<f32> = merged.bias().expect("bias preserved").tensor().to_vec();
    assert_eq!(merged_bias, bias_v);
}
