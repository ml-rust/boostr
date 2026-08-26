//! Unit tests for the VoxCPM2 per-patch generation loop.
//!
//! In a sibling module rather than inline so `generate.rs` stays inside the
//! architecture-file size limit — the same split `minicpm4/decode.rs` uses.
//! Fixtures and helpers live in [`support`]; this file holds the `#[test]`
//! functions themselves.

mod support;

use super::*;
use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, HIDDEN_DIM, PATCH_SIZE, t};
use crate::model::audio::voxcpm::minicpm4::model::tests::HIDDEN;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;
use support::*;

/// The stop fixture must actually answer what it claims, in BOTH directions
/// — every guard test below is vacuous otherwise.
#[test]
fn stop_chain_answers_its_configured_class() {
    let (client, device) = cpu_setup();
    for stop in [true, false] {
        let fx = fixture(stop, &device);
        let hidden = Var::new(t(&[1, HIDDEN], 0.4, &device), false);
        let logits = fx.aux.stop(&client, &hidden).expect("stop");
        assert_eq!(
            stop_predicted(&client, &logits).expect("argmax"),
            stop,
            "stop chain built for {stop} answered the other way"
        );
    }
}

/// Trap 1: iteration 0 is conditioned on the ZERO text-pad patch, and
/// iteration 1 on the patch iteration 0 emitted — never on anything from the
/// reference audio.
#[test]
fn prefix_feat_cond_is_zero_then_the_previous_patch() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let mut st = state(&fx, &device);
    let opts = options(2, 8);

    assert_eq!(st.prefix_feat_cond.shape(), &[1, PATCH_SIZE, FEAT_DIM]);
    assert!(
        values(&st.prefix_feat_cond).iter().all(|v| *v == 0.0),
        "iteration 0 must be conditioned on zeros"
    );

    generator
        .step_with_noise(&client, &mut st, &noise(0.2, &device), &opts)
        .expect("step 0");
    let patch0 = values(&st.patches[0]);
    assert!(
        patch0.iter().any(|v| v.abs() > 1e-6),
        "degenerate patch would make the comparison below vacuous"
    );
    assert_eq!(
        values(&st.prefix_feat_cond),
        patch0,
        "iteration 1 must be conditioned on iteration 0's patch"
    );
}

/// Trap 2, the strictly-greater half that fires: with `min_len = 2` and a
/// stop token on every iteration, the break lands at `i = 3` — so exactly
/// `min_len + 2` patches come out.
#[test]
fn stop_guard_fires_one_past_min_len() {
    let (client, device) = cpu_setup();
    let fx = fixture(true, &device);
    let mut st = state(&fx, &device);
    let opts = options(2, 12);

    let outcome = fx
        .generator()
        .generate(&client, &mut st, &opts)
        .expect("generate");
    assert_eq!(outcome, GenerateOutcome::StopToken);
    assert_eq!(
        st.patches.len(),
        opts.min_len + 2,
        "the guard is `i > min_len`, so i = min_len + 1 is the first break"
    );
}

/// Trap 2, the half that must NOT fire: capped at `min_len + 1` patches, a
/// stop token on every iteration still cannot end the run, because `i` never
/// exceeds `min_len`. A `>=` guard would return `StopToken` here.
#[test]
fn stop_guard_never_fires_at_min_len() {
    let (client, device) = cpu_setup();
    let fx = fixture(true, &device);
    let mut st = state(&fx, &device);
    let opts = options(2, 3);

    let outcome = fx
        .generator()
        .generate(&client, &mut st, &opts)
        .expect("generate");
    assert_eq!(
        outcome,
        GenerateOutcome::MaxLen,
        "a stop token at i <= min_len must be ignored"
    );
    assert_eq!(st.patches.len(), 3);
}

/// Trap 3: the cap exit is distinguishable from a stop-token exit.
#[test]
fn max_len_exit_reports_max_len() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let mut st = state(&fx, &device);
    let opts = options(2, 5);

    let outcome = fx
        .generator()
        .generate(&client, &mut st, &opts)
        .expect("generate");
    assert_eq!(outcome, GenerateOutcome::MaxLen);
    assert_eq!(st.patches.len(), 5);
    assert_eq!(st.prefill.position, 5);
}

/// Trap 4: ONE counter, both caches. Each iteration advances `position` and
/// BOTH cache lengths by exactly one.
#[test]
fn position_advances_both_caches_in_lockstep() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let mut st = state(&fx, &device);
    let opts = options(2, 8);

    for expected in 1..=4 {
        let outcome = generator.step(&client, &mut st, &opts).expect("step");
        assert_eq!(outcome, StepOutcome::Continued);
        assert_eq!(st.prefill.position, expected);
        assert_eq!(st.prefill.base_cache.seq_len(), expected);
        assert_eq!(st.prefill.residual_cache.seq_len(), expected);
    }
}

/// Trap 4's guard rail: a `position` that no longer matches the caches is
/// rejected by `decode_step`, so a drift errors instead of rotating a query
/// at one position while filing its key at another.
#[test]
fn desynced_position_errors() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let mut st = state(&fx, &device);
    let opts = options(2, 8);

    generator.step(&client, &mut st, &opts).expect("step 0");
    assert_eq!(st.prefill.position, 1);

    // Deliberate drift: the caches hold 1 position, the counter claims 2.
    st.prefill.position += 1;
    assert!(
        generator.step(&client, &mut st, &opts).is_err(),
        "a position/cache drift must error, not corrupt the cache"
    );
}

/// Trap 5: the stop guard fires BEFORE step 6, so a stopped iteration leaves
/// the caches and `position` where the previous iteration left them, exactly
/// as the reference's `break` does.
#[test]
fn a_stopped_step_does_not_advance_the_caches() {
    let (client, device) = cpu_setup();
    let fx = fixture(true, &device);
    let generator = fx.generator();
    let mut st = state(&fx, &device);
    let opts = options(2, 12);

    assert_eq!(
        generator
            .generate(&client, &mut st, &opts)
            .expect("generate"),
        GenerateOutcome::StopToken
    );
    // 4 patches emitted, but the last one broke before stepping the LMs.
    assert_eq!(st.patches.len(), 4);
    assert_eq!(st.prefill.position, 3);
    assert_eq!(st.prefill.base_cache.seq_len(), 3);
    assert_eq!(st.prefill.residual_cache.seq_len(), 3);
}

/// The injected-noise path is the primitive: the same `z` against the same
/// state reproduces the same patch bit for bit, which is what makes the CFM
/// gate possible.
#[test]
fn injected_noise_is_reproducible_and_load_bearing() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let opts = options(2, 8);

    let run = |z: &Var<CpuRuntime>| {
        let mut st = state(&fx, &device);
        generator
            .step_with_noise(&client, &mut st, z, &opts)
            .expect("step");
        values(&st.patches[0])
    };

    let a = run(&noise(0.2, &device));
    assert_eq!(a, run(&noise(0.2, &device)), "same z must give same patch");
    assert_ne!(
        a,
        run(&noise(4.6, &device)),
        "a different z must change the patch, or the noise is being ignored"
    );
}

/// Shape validation, at the two inputs a caller drives directly.
#[test]
fn rejects_wrong_shapes_and_a_zero_cap() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let mut st = state(&fx, &device);
    let opts = options(2, 8);

    // `z` is [1, feat_dim, patch_size], NOT the patch layout.
    let transposed = Var::new(t(&[1, PATCH_SIZE, FEAT_DIM], 0.5, &device), false);
    assert!(
        generator
            .step_with_noise(&client, &mut st, &transposed, &opts)
            .is_err()
    );
    assert!(
        generator
            .generate(&client, &mut st, &options(2, 0))
            .is_err()
    );
}

/// The tiny fixture must be able to run a whole clone-shaped loop: the
/// options constructor's real settings, not the test's cheap ones.
#[test]
fn default_options_carry_the_clone_scripts_values() {
    let opts = GenerateOptions::new(600, 0);
    assert_eq!(opts.cfm.n_timesteps, 32, "the clone script overrides 10");
    assert_eq!(opts.cfm.cfg_value, 2.0);
    assert_eq!(opts.min_len, 2);
    assert_eq!(opts.max_len, 600);
}

/// A rank-2 client sanity check on the fixture wiring: the loop's mu is two
/// DiT tokens wide, which is what `check_mu` derives.
#[test]
fn mu_is_two_dit_tokens_wide() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let hidden = Var::new(t(&[1, HIDDEN], 0.4, &device), false);
    let lm = fx
        .aux
        .lm_to_dit_proj
        .forward(&client, &hidden)
        .expect("lm_to_dit");
    let res = fx
        .aux
        .res_to_dit_proj
        .forward(&client, &hidden)
        .expect("res_to_dit");
    assert_eq!(lm.shape()[1] + res.shape()[1], 2 * HIDDEN_DIM);
}

/// Capturing must not perturb step 1-8 arithmetic. Guard OPEN (`i >
/// min_len`), so `aux.stop` runs on both paths here — the closed-guard
/// asymmetry is covered separately below.
#[test]
fn capturing_and_non_capturing_paths_agree() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let opts = options(0, 8);

    let mut plain = state(&fx, &device);
    let outcome = generator
        .step_with_noise(&client, &mut plain, &noise(0.2, &device), &opts)
        .expect("plain step");

    let mut captured = state(&fx, &device);
    let (captured_outcome, intermediates) = generator
        .step_with_noise_capturing(&client, &mut captured, &noise(0.2, &device), &opts)
        .expect("capturing step");

    assert_eq!(outcome, captured_outcome);
    assert_eq!(values(&plain.patches[0]), values(&captured.patches[0]));
    assert_eq!(
        values(&plain.prefill.lm_hidden),
        values(&captured.prefill.lm_hidden)
    );
    assert_eq!(
        values(&plain.prefill.residual_hidden),
        values(&captured.prefill.residual_hidden)
    );
    assert_eq!(plain.prefill.position, captured.prefill.position);

    // Non-degenerate: this must not pass by both sides silently zeroing out.
    assert!(values(&intermediates.mu).iter().any(|v| v.abs() > 1e-6));
    assert!(
        values(&intermediates.curr_embed)
            .iter()
            .any(|v| v.abs() > 1e-6)
    );
    assert!(
        intermediates.lm_hidden_pre_fsq.is_some(),
        "the guard was open, so steps 6-8 ran and lm_hidden_pre_fsq must be captured"
    );
    assert_eq!(intermediates.stop_logits.shape(), &[1, 2]);
}

/// Guard CLOSED (`i <= min_len`): the non-capturing path skips `aux.stop`
/// entirely, but capturing must still populate `stop_logits`, without
/// changing any other output.
#[test]
fn capturing_computes_stop_logits_even_when_the_guard_is_closed() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let opts = options(2, 8);

    let mut plain = state(&fx, &device);
    generator
        .step_with_noise(&client, &mut plain, &noise(0.2, &device), &opts)
        .expect("plain step");

    let mut captured = state(&fx, &device);
    let (outcome, intermediates) = generator
        .step_with_noise_capturing(&client, &mut captured, &noise(0.2, &device), &opts)
        .expect("capturing step");

    assert_eq!(
        outcome,
        StepOutcome::Continued,
        "i = 0 is below min_len = 2"
    );
    assert_eq!(intermediates.stop_logits.shape(), &[1, 2]);
    assert_eq!(values(&plain.patches[0]), values(&captured.patches[0]));
    assert_eq!(
        values(&plain.prefill.lm_hidden),
        values(&captured.prefill.lm_hidden)
    );
}
