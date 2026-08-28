//! Unit tests for [`super`]'s CFM training loss.
//!
//! Reuses `generate/tests/support.rs`'s `Fixture` — the exact same tiny
//! sub-models the teacher-forced tests exercise — rather than building a
//! second fixture. That module's items are `pub(crate)` specifically so
//! this sibling of `generate` can reach them directly.
//!
//! # Why the projections adapted differ between the gradient test and the
//! optimizer-loop test
//!
//! [`gradients_reach_every_lora_adapter`] adapts through
//! [`crate::nn::Module::trainable_parameters`], which works on any
//! sub-model uniformly (including per-layer attention/MLP projections
//! nested inside `Vec<layer>`), so it exercises the FULL fixture.
//!
//! [`cfm_loss_decreases_when_overfitting_one_batch`] additionally has to
//! WRITE optimizer-updated values back into the live `Fixture` between
//! steps (see [`crate::nn::maybe_lora::MaybeLoraLinear::set_adapters_with_ids`]),
//! which needs a direct, non-nested field path to each adapted projection.
//! It is restricted to `feat_decoder`'s three top-level projections and
//! `aux`'s four CFM-relevant ones (`enc_to_lm_proj`, `lm_to_dit_proj`,
//! `res_to_dit_proj`, `fusion_concat_proj` — `stop_proj`/`stop_head` are
//! deliberately excluded: the stop classifier is not on the CFM loss's
//! dependency graph at all, so adapting it would starve those adapters of
//! any gradient and the loop would have nothing to update).

use super::*;
use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, state};
use crate::nn::{LoraTargets, MaybeLoraLinear, Module};
use crate::test_utils::cpu_setup;
use crate::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::backward;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::TensorId;
use std::collections::HashMap;

/// `T = 3` — enough that the teacher-forced shift actually reads a batched
/// row (see `teacher_forced.rs`'s module docs), so `base_lm`, `residual_lm`,
/// `fsq` and `enc_to_lm_proj`/`fusion_concat_proj` all sit on a live
/// gradient path instead of computing an output nothing downstream reads.
const T: usize = 3;

fn target_patches(seed: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    t(&[T, PATCH_SIZE, FEAT_DIM], seed, device)
}

#[test]
fn cfm_loss_with_noise_is_finite_and_positive() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.4, &device);
    let noise = target_patches(1.7, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let loss = generator
        .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise)
        .expect("cfm_loss_with_noise");
    let val = loss.tensor().to_vec::<f32>()[0];
    assert!(val.is_finite(), "loss must be finite, got {val}");
    assert!(
        val > 0.0,
        "loss must be positive for mismatched prediction/target, got {val}"
    );
}

#[test]
fn cfm_loss_seeded_wrapper_is_finite_and_positive() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(2.3, &device);
    let loss = generator
        .cfm_loss(&client, &st.prefill, &target, 42)
        .expect("cfm_loss");
    let val = loss.tensor().to_vec::<f32>()[0];
    assert!(val.is_finite(), "loss must be finite, got {val}");
    assert!(val > 0.0, "loss must be positive, got {val}");
}

/// The single most important test in this unit: every LoRA adapter's
/// gradient must be present AND non-zero. A missing or all-zero adapter
/// gradient means the graph is severed somewhere and training silently
/// does nothing.
#[test]
fn gradients_reach_every_lora_adapter() {
    let (client, device) = cpu_setup();
    let mut fx = fixture(false, &device);

    let rank = 2;
    let alpha = 4.0;
    // `LocalEncoder` has no `cond_proj`/`out_proj` of its own (only
    // `in_proj` plus its layers' attention/MLP) — `apply_lora` is an ENTRY
    // POINT here (see its own doc comment) and requires EVERY target name
    // to match something in the tree it is called on, so this list is
    // deliberately narrower than `dit_targets` below rather than shared.
    let encoder_targets = LoraTargets::new([
        "in_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]);
    fx.feat_encoder
        .apply_lora(&encoder_targets, rank, alpha, &device, "feat_encoder")
        .expect("apply_lora feat_encoder");
    let dit_targets = LoraTargets::new([
        "in_proj",
        "cond_proj",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]);
    fx.feat_decoder
        .apply_lora(&dit_targets, rank, alpha, &device, "feat_decoder")
        .expect("apply_lora feat_decoder");
    let lm_targets = LoraTargets::new([
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]);
    fx.base_lm
        .apply_lora(&lm_targets, rank, alpha, &device, "base_lm")
        .expect("apply_lora base_lm");
    fx.residual_lm
        .apply_lora(&lm_targets, rank, alpha, &device, "residual_lm")
        .expect("apply_lora residual_lm");
    let fsq_targets = LoraTargets::new(["in_proj", "out_proj"]);
    fx.fsq
        .apply_lora(&fsq_targets, rank, alpha, &device, "fsq")
        .expect("apply_lora fsq");
    // `stop_proj`/`stop_head` excluded — see this module's doc comment.
    let aux_targets = LoraTargets::new([
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
        "fusion_concat_proj",
    ]);
    fx.aux
        .apply_lora(&aux_targets, rank, alpha, &device, "")
        .expect("apply_lora aux");

    let st = state(&fx, &device);
    let generator = fx.generator();

    // Named, so `lora_a` and `lora_b` can be told apart — they behave
    // differently at initialisation and conflating them hides a real bug
    // behind an expected zero. See the assertions below.
    let mut adapters: Vec<(String, TensorId)> = Vec::new();
    for (prefix, named) in [
        ("feat_encoder", Module::named_parameters(&fx.feat_encoder)),
        ("feat_decoder", Module::named_parameters(&fx.feat_decoder)),
        ("base_lm", Module::named_parameters(&fx.base_lm)),
        ("residual_lm", Module::named_parameters(&fx.residual_lm)),
        ("fsq", Module::named_parameters(&fx.fsq)),
        ("aux", Module::named_parameters(&fx.aux)),
    ] {
        for (name, var) in named {
            if var.requires_grad() && (name.ends_with("lora_a") || name.ends_with("lora_b")) {
                adapters.push((format!("{prefix}.{name}"), var.id()));
            }
        }
    }
    assert!(
        !adapters.is_empty(),
        "the target lists above must have matched at least one projection per sub-model"
    );

    let target = target_patches(0.6, &device);
    let loss = generator
        .cfm_loss(&client, &st.prefill, &target, 99)
        .expect("cfm_loss");
    let grads = backward(&loss, &client).expect("backward");

    let mut nonzero_b = 0usize;
    for (name, id) in &adapters {
        // EVERY adapter must at least APPEAR in the grad store. A missing
        // entry means the autograd graph never reached it, which is the
        // severed-graph failure this test exists to catch.
        let grad = grads
            .get(*id)
            .unwrap_or_else(|| panic!("adapter {name} has no gradient at all — graph is severed"));
        let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
        let any_nonzero = values.iter().any(|v| *v != 0.0);

        if name.ends_with("lora_b") {
            // `lora_b` carries signal immediately.
            assert!(
                any_nonzero,
                "adapter {name} has an all-zero gradient — graph reaches it but carries no signal"
            );
            nonzero_b += 1;
        } else {
            // `lora_a`'s gradient is EXACTLY ZERO at initialisation, and that
            // is correct, not a severed graph. The adapter path is
            // `(x @ A^T) @ B^T * scaling`, so `dL/dA` is proportional to `B`,
            // and standard LoRA init sets `B = 0` so the adapter starts as a
            // no-op. `A` only begins to move once `B` has. Asserting nonzero
            // here would be asserting that LoRA is initialised wrongly.
            assert!(
                !any_nonzero,
                "adapter {name} has a nonzero gradient at init, but dL/dA is \
                 proportional to B and B starts at zero — either the init or \
                 the adapter math changed"
            );
        }
    }
    assert!(nonzero_b > 0, "no lora_b adapter was checked");
}

/// Overwrite an adapted projection's `lora_a`/`lora_b` from the trainer's
/// updated `params` map, preserving their stable [`TensorId`]s. See this
/// module's doc comment for why only `feat_decoder`'s and `aux`'s
/// TOP-LEVEL projections are used in this test.
fn write_back(
    proj: &mut MaybeLoraLinear<CpuRuntime>,
    params: &HashMap<TensorId, Tensor<CpuRuntime>>,
) {
    let (a_id, b_id) = {
        let (a, b) = proj
            .adapters()
            .expect("projection must already be LoRA-adapted");
        (a.id(), b.id())
    };
    let a_tensor = params
        .get(&a_id)
        .expect("trainer must have updated lora_a")
        .clone();
    let b_tensor = params
        .get(&b_id)
        .expect("trainer must have updated lora_b")
        .clone();
    proj.set_adapters_with_ids(a_tensor, a_id, b_tensor, b_id)
        .expect("set_adapters_with_ids on an adapted projection");
}

/// Overfit one FIXED batch: `t`/`noise`/`target_patches` never change
/// across iterations, so the target is stationary and a loss decrease is
/// meaningful rather than an artifact of a moving target.
#[test]
fn cfm_loss_decreases_when_overfitting_one_batch() {
    let (client, device) = cpu_setup();
    let mut fx = fixture(false, &device);

    let rank = 2;
    let alpha = 4.0;
    let dit_targets = LoraTargets::new(["in_proj", "cond_proj", "out_proj"]);
    fx.feat_decoder
        .apply_lora(&dit_targets, rank, alpha, &device, "feat_decoder")
        .expect("apply_lora feat_decoder");
    let aux_targets = LoraTargets::new([
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
        "fusion_concat_proj",
    ]);
    fx.aux
        .apply_lora(&aux_targets, rank, alpha, &device, "")
        .expect("apply_lora aux");

    let st = state(&fx, &device);
    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let mut trainable: Vec<(TensorId, &Var<CpuRuntime>)> = Vec::new();
    trainable.extend(fx.feat_decoder.in_proj.trainable_parameters());
    trainable.extend(fx.feat_decoder.cond_proj.trainable_parameters());
    trainable.extend(fx.feat_decoder.out_proj.trainable_parameters());
    trainable.extend(fx.aux.enc_to_lm_proj.trainable_parameters());
    trainable.extend(fx.aux.lm_to_dit_proj.trainable_parameters());
    trainable.extend(fx.aux.res_to_dit_proj.trainable_parameters());
    trainable.extend(fx.aux.fusion_concat_proj.trainable_parameters());
    assert!(!trainable.is_empty(), "expected at least one LoRA adapter");
    let mut params: HashMap<TensorId, Tensor<CpuRuntime>> = trainable
        .into_iter()
        .map(|(id, var)| (id, var.tensor().clone()))
        .collect();

    // Deliberately small and deliberately plain SGD-like AdamW settings —
    // this overfits a single fixed batch of `T = 3` patches through a
    // handful of rank-2 adapters, not a realistic training recipe. No
    // weight decay and no grad-norm clipping: both would fight the loss
    // straight down on a fixture this small.
    let lr = 0.05;
    let config = TrainingConfig::default()
        .with_lr(lr)
        .with_weight_decay(0.0)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let mut first_loss = 0.0f64;
    let mut last_loss = 0.0f64;
    for step in 0..20 {
        let loss = {
            let generator = fx.generator();
            generator
                .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise)
                .expect("cfm_loss_with_noise")
        };
        let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        let grads = backward(&loss, &client).expect("backward");
        let metrics = trainer
            .step(&client, &mut params, grads, loss_val)
            .expect("trainer step");
        assert!(
            metrics.is_some(),
            "grad_accum_steps is 1, every step must finalize"
        );

        write_back(&mut fx.feat_decoder.in_proj, &params);
        write_back(&mut fx.feat_decoder.cond_proj, &params);
        write_back(&mut fx.feat_decoder.out_proj, &params);
        write_back(&mut fx.aux.enc_to_lm_proj, &params);
        write_back(&mut fx.aux.lm_to_dit_proj, &params);
        write_back(&mut fx.aux.res_to_dit_proj, &params);
        write_back(&mut fx.aux.fusion_concat_proj, &params);
    }

    assert!(
        last_loss < first_loss * 0.5,
        "loss should decrease meaningfully overfitting one fixed batch: first={first_loss} last={last_loss}"
    );
}

#[test]
fn cfm_loss_with_noise_rejects_bad_shapes() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let bad_target =
        Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE + 1, FEAT_DIM], DType::F32, &device)
            .expect("zeros");
    assert!(
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &bad_target, &ts, &noise)
            .is_err(),
        "a shape-mismatched target_patches must error, not panic"
    );

    let bad_t = Tensor::<CpuRuntime>::zeros(&[T + 1], DType::F32, &device).expect("zeros");
    assert!(
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &bad_t, &noise)
            .is_err(),
        "a shape-mismatched t must error, not panic"
    );

    let bad_noise =
        Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE, FEAT_DIM + 1], DType::F32, &device)
            .expect("zeros");
    assert!(
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &bad_noise)
            .is_err(),
        "a shape-mismatched noise must error, not panic"
    );

    let empty = Tensor::<CpuRuntime>::zeros(&[0, PATCH_SIZE, FEAT_DIM], DType::F32, &device)
        .expect("zeros");
    let empty_t = Tensor::<CpuRuntime>::zeros(&[0], DType::F32, &device).expect("zeros");
    let empty_noise = Tensor::<CpuRuntime>::zeros(&[0, PATCH_SIZE, FEAT_DIM], DType::F32, &device)
        .expect("zeros");
    assert!(
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &empty, &empty_t, &empty_noise)
            .is_err(),
        "T = 0 must error, not panic"
    );
}

#[test]
fn cfm_loss_seeded_wrapper_rejects_bad_target_shape() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let bad = Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE, FEAT_DIM], DType::F32, &device)
        .expect("zeros")
        .reshape(&[T * PATCH_SIZE, FEAT_DIM])
        .expect("reshape to wrong rank");
    assert!(
        generator.cfm_loss(&client, &st.prefill, &bad, 7).is_err(),
        "a rank-2 target_patches must error, not panic"
    );
}
