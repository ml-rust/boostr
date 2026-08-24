use super::*;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn test_lora_linear_creation() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight: Tensor<CpuRuntime> = Tensor::zeros(&[64, 32], DType::F32, &device).unwrap();
    let base = Linear::new(weight, None, false);
    let lora = LoraLinear::new(base, 8, 16.0, &device).expect("lora new must succeed on CPU");
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
        Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device).unwrap(),
        None,
        false,
    );
    let lora = LoraLinear::new(base, rank, 16.0, &device).expect("lora new must succeed on CPU");

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32) * 0.25 - 0.75)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
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
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
    let base = Linear::new(weight, Some(bias), true);
    let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

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
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
    let base = Linear::new(weight, None, false); // frozen
    let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

    let trainable = lora.trainable_parameters();
    assert_eq!(trainable.len(), 2);
    assert_eq!(trainable[0].0, lora.lora_a.id());
    assert_eq!(trainable[1].0, lora.lora_b.id());
}

/// `with_ids` must preserve BOTH supplied `TensorId`s exactly — this is its
/// entire reason to exist over `from_weights`. A resumed run rebuilds
/// `LoraLinear` from a `TensorId`-keyed optimizer-state map each step; if the
/// ids drifted, the rebuilt adapter would detach from its optimizer state.
#[test]
fn test_with_ids_preserves_supplied_ids() {
    let device = <CpuRuntime as Runtime>::default_device();
    let make_base = || {
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
        Linear::new(weight, None, false)
    };
    let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();
    let a_id = TensorId::new();
    let b_id = TensorId::new();

    let lora = LoraLinear::with_ids(make_base(), a, a_id, b, b_id, 4.0, true);

    assert_eq!(lora.lora_a().id(), a_id);
    assert_eq!(lora.lora_b().id(), b_id);
}

/// Contrast case: a resumed run reads adapter tensors OUT of a
/// `TensorId`-keyed optimizer-state map by reference, so it must `.clone()`
/// before handing them to `from_weights` — and `Tensor::clone` mints a FRESH
/// `TensorId` (`numr::tensor::core::Tensor::clone`, confirmed by reading its
/// impl: `Self { id: TensorId::new(), .. }`). This is precisely the
/// detachment `with_ids` exists to avoid: rebuilding via `from_weights` from
/// a stored map hands the adapter a new id every step. Proves the two
/// constructors are NOT interchangeable for that caller.
#[test]
fn test_from_weights_mints_fresh_ids_unlike_with_ids() {
    let device = <CpuRuntime as Runtime>::default_device();
    let make_base = || {
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
        Linear::new(weight, None, false)
    };
    // Simulates tensors held in a `TensorId`-keyed map, accessed by reference.
    let stored_a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
    let stored_b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();
    let stored_a_id = stored_a.id();
    let stored_b_id = stored_b.id();

    let lora = LoraLinear::from_weights(make_base(), stored_a.clone(), stored_b.clone(), 4.0, true);

    assert_ne!(
        lora.lora_a().id(),
        stored_a_id,
        "from_weights unexpectedly preserved the cloned lora_a id — the contrast \
         case no longer holds and with_ids may be redundant"
    );
    assert_ne!(
        lora.lora_b().id(),
        stored_b_id,
        "from_weights unexpectedly preserved the cloned lora_b id — the contrast \
         case no longer holds and with_ids may be redundant"
    );
}

/// `trainable` must apply to both factors, in both directions, via `with_ids`.
#[test]
fn test_with_ids_trainable_flag() {
    let device = <CpuRuntime as Runtime>::default_device();
    let make_base = || {
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
        Linear::new(weight, None, false)
    };
    let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();

    let frozen = LoraLinear::with_ids(
        make_base(),
        a.clone(),
        TensorId::new(),
        b.clone(),
        TensorId::new(),
        4.0,
        false,
    );
    assert!(!frozen.lora_a().requires_grad());
    assert!(!frozen.lora_b().requires_grad());

    let trainable = LoraLinear::with_ids(
        make_base(),
        a,
        TensorId::new(),
        b,
        TensorId::new(),
        4.0,
        true,
    );
    assert!(trainable.lora_a().requires_grad());
    assert!(trainable.lora_b().requires_grad());
}

/// `rank()` and `scaling()` must derive from the supplied tensors/`alpha`,
/// not from any assumption baked into `LoraLinear::new`'s init path.
#[test]
fn test_with_ids_derives_rank_and_scaling() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 20], &[4, 5], &device).unwrap();
    let base = Linear::new(weight, None, false);
    let (rank, in_features, out_features) = (5usize, 5usize, 4usize);
    let a = Tensor::<CpuRuntime>::from_slice(
        &vec![0.1f32; rank * in_features],
        &[rank, in_features],
        &device,
    )
    .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(
        &vec![0.2f32; out_features * rank],
        &[out_features, rank],
        &device,
    )
    .unwrap();

    let lora = LoraLinear::with_ids(base, a, TensorId::new(), b, TensorId::new(), 15.0, true);

    assert_eq!(lora.rank(), rank);
    assert!((lora.scaling() - 3.0).abs() < 1e-6); // alpha/rank = 15/5 = 3
}

/// Ids must not affect numerics: a `with_ids`-built layer and an equivalent
/// `from_weights`-built layer must produce identical forward output.
#[test]
fn test_with_ids_forward_matches_from_weights() {
    use crate::test_utils::cpu_setup;

    let (client, device) = cpu_setup();
    let (in_features, out_features, rank) = (3usize, 2usize, 2usize);

    let base_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32) * 0.1 - 0.3)
        .collect();
    let make_base = || {
        Linear::new(
            Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device)
                .unwrap(),
            None,
            false,
        )
    };

    let a_vals: Vec<f32> = (0..rank * in_features)
        .map(|i| (i as f32) * 0.05 - 0.1)
        .collect();
    let b_vals: Vec<f32> = (0..out_features * rank)
        .map(|i| (i as f32) * 0.07 + 0.02)
        .collect();

    let with_ids_lora = LoraLinear::with_ids(
        make_base(),
        Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device).unwrap(),
        TensorId::new(),
        Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device).unwrap(),
        TensorId::new(),
        8.0,
        false,
    );
    let from_weights_lora = LoraLinear::from_weights(
        make_base(),
        Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device).unwrap(),
        Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device).unwrap(),
        8.0,
        false,
    );

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32) * 0.2 - 0.4)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
        false,
    );

    let with_ids_out = with_ids_lora
        .forward(&client, &x)
        .expect("with_ids forward");
    let from_weights_out = from_weights_lora
        .forward(&client, &x)
        .expect("from_weights forward");

    let with_ids_vals: Vec<f32> = with_ids_out.tensor().contiguous().expect("contig").to_vec();
    let from_weights_vals: Vec<f32> = from_weights_out
        .tensor()
        .contiguous()
        .expect("contig")
        .to_vec();
    assert_eq!(with_ids_vals.len(), from_weights_vals.len());
    for (w, f) in with_ids_vals.iter().zip(from_weights_vals.iter()) {
        assert!((w - f).abs() < 1e-5, "with_ids={w} from_weights={f}");
    }
}

#[test]
fn test_from_weights_trainable_flag() {
    let device = <CpuRuntime as Runtime>::default_device();
    let make_base = || {
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
        Linear::new(weight, None, false)
    };
    let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();

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
        Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device).unwrap(),
        Some(Tensor::<CpuRuntime>::from_slice(&bias_v, &[out_features], &device).unwrap()),
        false,
    );

    let a_vals: Vec<f32> = (0..rank * in_features)
        .map(|i| (i as f32) * 0.05 - 0.1)
        .collect();
    // Deliberately non-zero, unlike LoraLinear::new's zero-init.
    let b_vals: Vec<f32> = (0..out_features * rank)
        .map(|i| (i as f32) * 0.07 + 0.02)
        .collect();
    let lora_a = Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device).unwrap();
    let lora_b = Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device).unwrap();
    let lora = LoraLinear::from_weights(base, lora_a, lora_b, 8.0, false);

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32) * 0.2 - 0.4)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
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
