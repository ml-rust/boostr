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
//!
//! # Stop-loss tests
//!
//! [`stop_loss_reaches_stop_head_but_diff_alone_does_not`] adapts
//! `stop_proj`/`stop_head` specifically to demonstrate the fact the doc
//! comment above states: `cfm_loss` alone leaves them with NO gradient
//! entry, and [`PatchGenerator::train_losses_with_noise`]'s `total` is what
//! puts them on the graph.

use super::*;
use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, state};
use crate::nn::{LoraTargets, MaybeLoraLinear, Module};
use crate::test_utils::cpu_setup;
use crate::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::backward;
use numr::ops::RandomOps;
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
        .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
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
        .cfm_loss(&client, &st.prefill, &target, 42, 0.0)
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
        .cfm_loss(&client, &st.prefill, &target, 99, 0.0)
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
                .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
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
            .cfm_loss_with_noise(&client, &st.prefill, &bad_target, &ts, &noise, false)
            .is_err(),
        "a shape-mismatched target_patches must error, not panic"
    );

    let bad_t = Tensor::<CpuRuntime>::zeros(&[T + 1], DType::F32, &device).expect("zeros");
    assert!(
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &bad_t, &noise, false)
            .is_err(),
        "a shape-mismatched t must error, not panic"
    );

    let bad_noise =
        Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE, FEAT_DIM + 1], DType::F32, &device)
            .expect("zeros");
    assert!(
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &bad_noise, false)
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
            .cfm_loss_with_noise(&client, &st.prefill, &empty, &empty_t, &empty_noise, false)
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
        generator
            .cfm_loss(&client, &st.prefill, &bad, 7, 0.0)
            .is_err(),
        "a rank-2 target_patches must error, not panic"
    );
}

#[test]
fn stop_loss_is_finite_and_positive() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.4, &device);
    let cond = generator
        .teacher_forced_conditioning(&client, &st.prefill, &target)
        .expect("teacher_forced_conditioning");
    let loss = generator.stop_loss(&client, &cond).expect("stop_loss");
    let val = loss.tensor().to_vec::<f32>()[0];
    assert!(val.is_finite(), "stop loss must be finite, got {val}");
    assert!(val > 0.0, "stop loss must be positive, got {val}");
}

/// Pins the target construction itself, not the model: `logits` are built
/// by hand, so this catches a wrong-position stop target (e.g. class 1 on
/// EVERY patch, or on patch 0) even if the model side of `stop_loss` is
/// correct.
#[test]
fn stop_target_is_the_last_patch_only() {
    let (client, device) = cpu_setup();

    // Confidently "continue" (class 0) on every one of the T = 3 patches,
    // including the last — the runaway-generation failure mode this loss
    // exists to penalise.
    #[rustfmt::skip]
    let all_continue = Var::new(
        Tensor::<CpuRuntime>::from_slice(
            &[5.0f32, -5.0,
              5.0, -5.0,
              5.0, -5.0],
            &[T, 2],
            &device,
        )
        .expect("logits"),
        false,
    );
    // Confidently "continue" for every patch except the last, "stop" on the
    // last — exactly the target `stop_targets` builds.
    #[rustfmt::skip]
    let stop_on_last = Var::new(
        Tensor::<CpuRuntime>::from_slice(
            &[5.0f32, -5.0,
              5.0, -5.0,
              -5.0, 5.0],
            &[T, 2],
            &device,
        )
        .expect("logits"),
        false,
    );

    let loss_all_continue =
        stop_loss_from_logits(&client, &all_continue).expect("stop_loss_from_logits");
    let loss_stop_on_last =
        stop_loss_from_logits(&client, &stop_on_last).expect("stop_loss_from_logits");
    let val_all_continue = loss_all_continue.tensor().to_vec::<f32>()[0];
    let val_stop_on_last = loss_stop_on_last.tensor().to_vec::<f32>()[0];

    assert!(
        val_all_continue > val_stop_on_last,
        "predicting \"continue\" on the LAST patch too must score worse than \
         predicting \"stop\" only there: all_continue={val_all_continue} \
         stop_on_last={val_stop_on_last}"
    );
    assert!(
        val_stop_on_last < 0.1,
        "a confident, correctly-placed stop prediction should be near zero, \
         got {val_stop_on_last}"
    );
}

/// The finding this whole unit exists to fix: `cfm_loss` alone never calls
/// `aux.stop`, so `stop_proj`/`stop_head` get NO gradient entry from it —
/// and `train_losses_with_noise`'s `total` is what puts them on the graph.
#[test]
fn stop_loss_reaches_stop_head_but_diff_alone_does_not() {
    let (client, device) = cpu_setup();
    let mut fx = fixture(false, &device);

    let rank = 2;
    let alpha = 4.0;
    // Adapt the stop chain AND one projection that IS inside the CFM graph.
    // Without the second, the diffusion loss would have no trainable input at
    // all and `backward` would fail outright rather than returning an empty
    // gradient — which proves the same point, but by erroring instead of by
    // measuring. Adapting `lm_to_dit_proj` too keeps the diff-only backward
    // well-formed, so "the stop head got nothing" is an OBSERVATION rather
    // than an exception.
    let stop_targets_list = LoraTargets::new(["stop_proj", "stop_head"]);
    fx.aux
        .apply_lora(&stop_targets_list, rank, alpha, &device, "")
        .expect("apply_lora aux stop chain");
    fx.aux
        .apply_lora(
            &LoraTargets::new(["lm_to_dit_proj"]),
            rank,
            alpha,
            &device,
            "",
        )
        .expect("apply_lora aux lm_to_dit_proj");

    // Only the STOP-chain adapters are under test; `lm_to_dit_proj`'s exist
    // solely to keep the diff-only graph alive.
    let mut adapters: Vec<(String, TensorId)> = Vec::new();
    for (name, var) in Module::named_parameters(&fx.aux) {
        let is_stop_chain = name.contains("stop_proj") || name.contains("stop_head");
        if var.requires_grad()
            && is_stop_chain
            && (name.ends_with("lora_a") || name.ends_with("lora_b"))
        {
            adapters.push((name, var.id()));
        }
    }
    assert!(
        !adapters.is_empty(),
        "stop_proj/stop_head must have matched the target list above"
    );

    let st = state(&fx, &device);
    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    // The CFM loss ALONE: the stop head sits outside its graph, so every
    // adapter above must be either absent from the grad store or all-zero.
    let diff_only = {
        let generator = fx.generator();
        generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
            .expect("cfm_loss_with_noise")
    };
    let diff_grads = backward(&diff_only, &client).expect("backward diff_only");
    for (name, id) in &adapters {
        let has_signal = diff_grads
            .get(*id)
            .map(|grad| {
                let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
                values.iter().any(|v| *v != 0.0)
            })
            .unwrap_or(false);
        assert!(
            !has_signal,
            "adapter {name} got a nonzero gradient from cfm_loss ALONE — the \
             stop head should be off that graph entirely"
        );
    }

    // The combined loss: `total` must reach every one of those adapters
    // with a nonzero gradient.
    let losses = {
        let generator = fx.generator();
        generator
            .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 1.0, false)
            .expect("train_losses_with_noise")
    };
    let total_grads = backward(&losses.total, &client).expect("backward total");
    let mut nonzero_b = 0usize;
    for (name, id) in &adapters {
        let grad = total_grads.get(*id).unwrap_or_else(|| {
            panic!("adapter {name} has no gradient at all from the combined loss")
        });
        let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
        let any_nonzero = values.iter().any(|v| *v != 0.0);
        if name.ends_with("lora_b") {
            assert!(
                any_nonzero,
                "adapter {name} has an all-zero gradient from the combined loss"
            );
            nonzero_b += 1;
        }
        // `lora_a`'s gradient is exactly zero at LoRA init regardless of
        // which loss produced it (dL/dA is proportional to B, and B starts
        // at zero) — see `gradients_reach_every_lora_adapter`'s identical
        // reasoning. Only `lora_b` is asserted nonzero here.
    }
    assert!(nonzero_b > 0, "no stop-chain lora_b adapter was checked");
}

/// `lambda_stop = 0.0` must make `total` numerically equal `lambda_diff *
/// diff` — the same value `cfm_loss_with_noise` returns on its own, since
/// `lambda_diff = 1.0` here.
#[test]
fn lambda_stop_zero_matches_diff_loss_alone() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let diff_alone = generator
        .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
        .expect("cfm_loss_with_noise");
    let diff_alone_val = diff_alone.tensor().to_vec::<f32>()[0];

    let losses = generator
        .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 0.0, false)
        .expect("train_losses_with_noise");
    let total_val = losses.total.tensor().to_vec::<f32>()[0];
    let diff_val = losses.diff.tensor().to_vec::<f32>()[0];

    assert!(
        (total_val - diff_alone_val).abs() < 1e-5,
        "lambda_stop = 0.0 must leave total == diff alone: total={total_val} \
         diff_alone={diff_alone_val}"
    );
    assert!(
        (diff_val - diff_alone_val).abs() < 1e-5,
        "TrainLosses::diff must equal cfm_loss_with_noise's own value: \
         diff={diff_val} diff_alone={diff_alone_val}"
    );
}

/// `drop_cond = true` must produce a finite loss that DIFFERS from
/// `drop_cond = false` on the identical `t`/`noise` — otherwise the
/// dropout parameter is wired in but never actually changes anything.
#[test]
fn drop_cond_true_changes_the_diff_loss() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let loss_kept = generator
        .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
        .expect("cfm_loss_with_noise drop_cond=false");
    let loss_dropped = generator
        .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, true)
        .expect("cfm_loss_with_noise drop_cond=true");

    let val_kept = loss_kept.tensor().to_vec::<f32>()[0];
    let val_dropped = loss_dropped.tensor().to_vec::<f32>()[0];
    assert!(
        val_dropped.is_finite(),
        "drop_cond=true loss must be finite, got {val_dropped}"
    );
    assert!(
        (val_dropped - val_kept).abs() > 1e-6,
        "drop_cond=true must change the diff loss: kept={val_kept} dropped={val_dropped}"
    );
}

/// `drop_cond = false` must be bit-identical to the pre-dropout behaviour:
/// [`PatchGenerator::teacher_forced_conditioning`] followed by
/// [`PatchGenerator::cfm_loss_from_conditioning`] directly, with no dropout
/// applied anywhere. Pins that the default path is untouched by this unit.
#[test]
fn drop_cond_false_matches_conditioning_computed_directly() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let via_wrapper = generator
        .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
        .expect("cfm_loss_with_noise");

    let cond = generator
        .teacher_forced_conditioning(&client, &st.prefill, &target)
        .expect("teacher_forced_conditioning");
    let direct = generator
        .cfm_loss_from_conditioning(&client, &cond, &target, &ts, &noise, T)
        .expect("cfm_loss_from_conditioning");

    let val_wrapper = via_wrapper.tensor().to_vec::<f32>()[0];
    let val_direct = direct.tensor().to_vec::<f32>()[0];
    assert_eq!(
        val_wrapper.to_bits(),
        val_direct.to_bits(),
        "drop_cond=false must be bit-identical to the undropped direct computation: \
         wrapper={val_wrapper} direct={val_direct}"
    );
}

/// The stop loss reads `cond.lm_hidden`, not `cond.mu`, so `drop_cond` must
/// leave it numerically IDENTICAL — the dropout must not leak into the stop
/// term.
#[test]
fn drop_cond_does_not_change_the_stop_loss() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);
    let noise = target_patches(1.9, &device);
    let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

    let kept = generator
        .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 1.0, false)
        .expect("train_losses_with_noise drop_cond=false");
    let dropped = generator
        .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 1.0, true)
        .expect("train_losses_with_noise drop_cond=true");

    let stop_kept = kept.stop.tensor().to_vec::<f32>()[0];
    let stop_dropped = dropped.stop.tensor().to_vec::<f32>()[0];
    assert_eq!(
        stop_kept.to_bits(),
        stop_dropped.to_bits(),
        "loss/stop must be identical regardless of drop_cond: kept={stop_kept} \
         dropped={stop_dropped}"
    );
}

/// `training_cfg_rate = 0.0` must never drop and `1.0` must always drop,
/// checked through the seeded wrapper across several seeds by comparing
/// against the deterministic form with the matching `t`/`noise` draws and
/// an explicit `drop_cond`.
#[test]
fn training_cfg_rate_boundaries_are_deterministic() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);
    let dtype = target.dtype();
    let shape = target.shape();

    for seed in [1u64, 42, 100, 9_999] {
        let t_draw = client
            .rand_seeded(&[T], dtype, seed)
            .expect("rand_seeded t");
        let noise_draw = client
            .randn_seeded(shape, dtype, seed.wrapping_add(1))
            .expect("randn_seeded noise");

        let never_dropped = generator
            .cfm_loss(&client, &st.prefill, &target, seed, 0.0)
            .expect("cfm_loss rate=0.0");
        let never_dropped_direct = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &t_draw, &noise_draw, false)
            .expect("cfm_loss_with_noise drop_cond=false");
        assert_eq!(
            never_dropped.tensor().to_vec::<f32>()[0].to_bits(),
            never_dropped_direct.tensor().to_vec::<f32>()[0].to_bits(),
            "training_cfg_rate=0.0 must never drop (seed={seed})"
        );

        let always_dropped = generator
            .cfm_loss(&client, &st.prefill, &target, seed, 1.0)
            .expect("cfm_loss rate=1.0");
        let always_dropped_direct = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &t_draw, &noise_draw, true)
            .expect("cfm_loss_with_noise drop_cond=true");
        assert_eq!(
            always_dropped.tensor().to_vec::<f32>()[0].to_bits(),
            always_dropped_direct.tensor().to_vec::<f32>()[0].to_bits(),
            "training_cfg_rate=1.0 must always drop (seed={seed})"
        );
    }
}

/// An out-of-range `training_cfg_rate` must error, not panic — from both
/// [`PatchGenerator::cfm_loss`] and [`PatchGenerator::train_losses`].
#[test]
fn training_cfg_rate_out_of_range_errors() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let target = target_patches(0.6, &device);

    for bad_rate in [-0.1, 1.5] {
        assert!(
            generator
                .cfm_loss(&client, &st.prefill, &target, 1, bad_rate)
                .is_err(),
            "cfm_loss must reject training_cfg_rate={bad_rate}, not panic"
        );
        assert!(
            generator
                .train_losses(&client, &st.prefill, &target, 1, 1.0, 1.0, bad_rate)
                .is_err(),
            "train_losses must reject training_cfg_rate={bad_rate}, not panic"
        );
    }
}
