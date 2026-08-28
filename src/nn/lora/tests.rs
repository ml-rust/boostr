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

/// `Module::parameters` is UNFILTERED: it reports the dense base's
/// weight/bias alongside the adapter factors, and the caller's
/// `requires_grad` filter decides what actually trains.
///
/// Reporting only the adapters would break a real caller. oxidizr's
/// `lora.train_modules` deliberately opts a named projection's base back
/// into training even under `freeze_base: true`; if `parameters()` dropped
/// it, that weight would be checkpointed every step and never once updated
/// by the optimizer — silently, with a healthy-looking run.
///
/// A QUANTIZED base needs no special handling here: it has no `Var` weight,
/// and `MaybeQuantLinear::parameters` is already empty for those variants,
/// so this same code reports adapters alone. See
/// `test_quantized_base_trainable_parameters_are_adapters_only`.
#[test]
fn test_module_parameters_reports_base_and_adapters() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
    let base = Linear::new(weight, Some(bias), true);
    let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

    // base.weight + base.bias + lora_a + lora_b.
    assert_eq!(lora.parameters().len(), 4);

    let named = lora.named_parameters();
    assert_eq!(named.len(), 4);
    assert!(named.iter().any(|(n, _)| n == "lora_a"));
    assert!(named.iter().any(|(n, _)| n == "lora_b"));
    assert!(named.iter().any(|(n, _)| n == "base.weight"));
    assert!(named.iter().any(|(n, _)| n == "base.bias"));
}

/// The regression the doc comment above describes, pinned directly: a dense
/// base left TRAINABLE must survive `trainable_parameters()`.
#[test]
fn test_trainable_base_is_not_dropped_from_trainable_parameters() {
    let device = <CpuRuntime as Runtime>::default_device();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
    let base = Linear::new(weight, None, true); // deliberately trainable
    let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

    // base.weight + lora_a + lora_b: the optimizer must be able to step all
    // three, or `lora.train_modules` silently stops working.
    assert_eq!(lora.trainable_parameters().len(), 3);
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

// --- QLoRA: adapter over a quantized base -----------------------------

/// Build a `LoraLinear` whose frozen base is a Q6_K block-quantized weight
/// (`in_features` must be a multiple of Q6_K's 256-element block size).
fn quantized_lora(
    client: &numr::runtime::cpu::CpuClient,
    device: &numr::runtime::cpu::CpuDevice,
    out_features: usize,
    in_features: usize,
    rank: usize,
    alpha: f32,
) -> LoraLinear<CpuRuntime> {
    use crate::nn::linear::{MaybeQuantLinear, QuantLinear};
    use crate::quant::format::QuantFormat;
    use crate::quant::traits::QuantizeOps;

    let base_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32 * 0.013).sin() * 0.3)
        .collect();
    let base_tensor =
        Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], device).unwrap();
    let quant = client
        .quantize(&base_tensor, QuantFormat::Q6K)
        .expect("Q6_K quantize");
    let base = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));

    LoraLinear::new(base, rank, alpha, device).expect("lora new must succeed over a quantized base")
}

/// A LoRA adapter over a `MaybeQuantLinear::Quantized` base must still
/// produce a finite, correctly-shaped forward output — this is the whole
/// point of QLoRA: fine-tune directly on a quantized checkpoint.
#[test]
fn test_lora_forward_over_quantized_base_is_finite_and_correct_shape() {
    use crate::test_utils::cpu_setup;

    let (client, device) = cpu_setup();
    let (out_features, in_features, rank) = (4usize, 256usize, 2usize);
    let lora = quantized_lora(&client, &device, out_features, in_features, rank, 8.0);

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32 * 0.004) - 1.0)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
        false,
    );

    let out = lora
        .forward(&client, &x)
        .expect("lora forward over quantized base");
    assert_eq!(out.shape(), &[2, out_features]);

    let vals: Vec<f32> = out.tensor().contiguous().expect("contig").to_vec();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "quantized-base LoRA output must be finite: {vals:?}"
    );
}

/// `trainable_parameters()` on a quantized-base adapter must report exactly
/// the two adapter tensors and nothing from the base — the base has no
/// `Var<R>` weight to report in the first place.
#[test]
fn test_quantized_base_trainable_parameters_are_adapters_only() {
    use crate::test_utils::cpu_setup;

    let (client, device) = cpu_setup();
    let lora = quantized_lora(&client, &device, 4, 256, 2, 8.0);

    let trainable = lora.trainable_parameters();
    assert_eq!(trainable.len(), 2);
    assert_eq!(trainable[0].0, lora.lora_a().id());
    assert_eq!(trainable[1].0, lora.lora_b().id());

    // The base itself has no `Var<R>` weight to have contributed one.
    assert!(lora.weight().is_none());

    let params = lora.parameters();
    assert_eq!(params.len(), 2);
}

/// Merging a LoRA adapter into a quantized base is not possible without
/// requantizing the merged result — `merge_into_base` must error, not panic,
/// and the error must say so explicitly.
#[test]
fn test_merge_into_base_errors_on_quantized_base() {
    use crate::test_utils::cpu_setup;

    let (client, device) = cpu_setup();
    let lora = quantized_lora(&client, &device, 4, 256, 2, 8.0);

    // `Linear<R>` is not `Debug`, so `expect_err` (which needs `T: Debug`)
    // cannot be used on this `Result<Linear<R>, _>`.
    let message = match lora.merge_into_base(&client) {
        Ok(_) => panic!("merging into a quantized base must fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("quantiz") && message.contains("requant"),
        "error must name the quantized base and the requantization requirement: {message}"
    );
}

/// The exact QLoRA arrangement the fix targets: a LoRA adapter over a
/// quantized base feeds a SECOND, downstream quantized projection before the
/// loss (`adapter -> quantized projection -> loss`). Before the fix, that
/// downstream `MaybeQuantLinear::Quantized::forward` detached the graph, so
/// `backward` reached neither this adapter nor anything upstream of it.
#[test]
fn test_lora_gradient_reaches_adapter_through_downstream_quantized_projection() {
    use crate::nn::linear::{MaybeQuantLinear, QuantLinear};
    use crate::quant::format::QuantFormat;
    use crate::quant::traits::QuantizeOps;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};

    let (client, device) = cpu_setup();
    // First layer's out_features (32) doubles as the second layer's
    // in_features, so it must satisfy Q8_0's 32-element block size too.
    let (in_features, mid_features, out_features, rank) = (256usize, 32usize, 4usize, 2usize);
    let lora = quantized_lora(&client, &device, mid_features, in_features, rank, 8.0);

    let second_w: Vec<f32> = (0..out_features * mid_features)
        .map(|i| (i as f32 * 0.021).cos() * 0.4)
        .collect();
    let second_tensor =
        Tensor::<CpuRuntime>::from_slice(&second_w, &[out_features, mid_features], &device)
            .unwrap();
    let second_quant = client
        .quantize(&second_tensor, QuantFormat::Q8_0)
        .expect("Q8_0 quantize");
    let second_layer = MaybeQuantLinear::Quantized(QuantLinear::new(second_quant, None));

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| (i as f32 * 0.005) - 0.5)
        .collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
        false,
    );

    let mid = lora.forward(&client, &x).expect("adapter forward");
    assert!(
        mid.requires_grad(),
        "adapter output must require grad (lora_a/lora_b are trainable)"
    );

    let out = second_layer
        .forward(&client, &mid)
        .expect("downstream quantized projection forward");
    assert!(
        out.requires_grad(),
        "downstream quantized projection must not detach the adapter's graph"
    );

    let loss = var_sum(&out, &[0, 1], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    assert!(
        grads.get(lora.lora_b().id()).is_some(),
        "lora_b must receive a gradient through the downstream quantized projection"
    );
    assert!(
        grads.get(lora.lora_a().id()).is_some(),
        "lora_a must receive a gradient through the downstream quantized projection"
    );
}
