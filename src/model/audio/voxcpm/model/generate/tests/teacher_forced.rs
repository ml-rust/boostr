//! Equivalence test for [`teacher_forced`](super::super::teacher_forced):
//! feeding the per-patch loop's OWN output back in as ground truth must
//! reproduce the loop's own `mu`/`curr_embed`, patch by patch, since the
//! only difference between the two paths is batched-vs-stepped evaluation
//! of the identical arithmetic. See `teacher_forced.rs`'s module docs for
//! why.

use super::super::*;
use super::support::*;
use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE};
use crate::model::audio::voxcpm::minicpm4::model::tests::HIDDEN;
use crate::test_utils::cpu_setup;
use numr::autograd::var_cat;
use numr::runtime::cpu::CpuRuntime;

/// Absolute tolerance for the batched-vs-stepped comparison. Both paths run
/// the SAME arithmetic — one full-sequence `forward`, one `decode_step` per
/// position — so any real discrepancy is a shift/concat-order bug, not
/// float noise.
///
/// MEASURED by bisection on this fixture: the real error sits between 1e-7
/// and 1e-6, so this is ~10x headroom over what was observed rather than an
/// unverified guess. It is deliberately NOT tightened to the observed value:
/// a larger fixture sums more terms per dot product and reassociates more,
/// so a bound fitted exactly to `HIDDEN = 8` would fail for reasons that are
/// not bugs.
///
/// The bound being tight is what makes this test load-bearing, and that was
/// CHECKED, not assumed: reversing the shift (`narrow(.., 1, t-1)` instead of
/// `narrow(.., 0, t-1)`) and swapping the fusion concat order to
/// `(curr_embed, lm_hidden)` each make
/// `teacher_forced_conditioning_reproduces_the_loop` FAIL. Those are the two
/// failure modes that no amount of reading the code catches.
const TOL: f32 = 1e-5;

fn assert_rows_close(actual: &[f32], expected_rows: &[Vec<f32>], what: &str) {
    let width = expected_rows.first().map_or(0, |r| r.len());
    assert_eq!(
        actual.len(),
        width * expected_rows.len(),
        "{what}: total length mismatch"
    );
    for (i, expected) in expected_rows.iter().enumerate() {
        let row = &actual[i * width..(i + 1) * width];
        for (j, (a, e)) in row.iter().zip(expected).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= TOL,
                "{what}[{i}][{j}]: batched {a} vs stepped {e} (diff {diff})"
            );
        }
    }
}

/// Steps the loop `n` times via the capturing path with distinct noise per
/// step, returning each step's `mu`/`curr_embed` (flattened) plus the
/// emitted patches, so the caller can feed those SAME patches into
/// [`PatchGenerator::teacher_forced_conditioning`] and compare.
fn run_steps(
    client: &numr::runtime::cpu::CpuClient,
    generator: &PatchGenerator<'_, CpuRuntime>,
    st: &mut GenerateState<CpuRuntime>,
    opts: &GenerateOptions,
    n: usize,
    device: &numr::runtime::cpu::CpuDevice,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut mus = Vec::with_capacity(n);
    let mut embeds = Vec::with_capacity(n);
    for i in 0..n {
        let (outcome, intermediates) = generator
            .step_with_noise_capturing(client, st, &noise(0.1 + i as f32 * 0.3, device), opts)
            .expect("step_with_noise_capturing");
        assert_eq!(
            outcome,
            StepOutcome::Continued,
            "min_len must outlast this run so every step is comparable"
        );
        mus.push(values(&intermediates.mu));
        embeds.push(values(&intermediates.curr_embed));
    }
    (mus, embeds)
}

/// The point of this unit: batched teacher forcing reproduces the stepped
/// loop exactly (up to float reassociation), for a multi-patch run.
#[test]
fn teacher_forced_conditioning_reproduces_the_loop() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();

    // Two independent, IDENTICAL prefill states: `support::state` is
    // deterministic (no randomness), so state B's `prefill` is exactly
    // state A's ORIGINAL prefill, before the loop below mutates it.
    let mut st_a = state(&fx, &device);
    let st_b = state(&fx, &device);
    let opts = options(100, 8); // min_len far past this run: guard never fires

    let n = 4;
    let (mus, embeds) = run_steps(&client, &generator, &mut st_a, &opts, n, &device);

    let patch_refs: Vec<&Var<CpuRuntime>> = st_a.patches.iter().collect();
    let target = var_cat(&patch_refs, 0, &client).expect("stack patches into [T, P, D]");

    let out = generator
        .teacher_forced_conditioning(&client, &st_b.prefill, target.tensor())
        .expect("teacher_forced_conditioning");

    assert_eq!(out.mu.shape(), &[n, 2 * HIDDEN]);
    assert_eq!(out.curr_embed.shape(), &[n, HIDDEN]);

    assert_rows_close(&values(&out.mu), &mus, "mu");
    assert_rows_close(&values(&out.curr_embed), &embeds, "curr_embed");
}

/// `T = 1`: only the zero-`cond` path runs, and no shift ever reads a
/// batched row (there isn't one) — `mu_0`/`curr_embed_0` must still match
/// the loop's own first step exactly.
#[test]
fn teacher_forced_conditioning_handles_a_single_patch() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();

    let mut st_a = state(&fx, &device);
    let st_b = state(&fx, &device);
    let opts = options(100, 8);

    let (mus, embeds) = run_steps(&client, &generator, &mut st_a, &opts, 1, &device);

    let target = st_a.patches[0].tensor().clone();
    let out = generator
        .teacher_forced_conditioning(&client, &st_b.prefill, &target)
        .expect("teacher_forced_conditioning");

    assert_eq!(out.mu.shape(), &[1, 2 * HIDDEN]);
    assert_eq!(out.cond.shape(), st_b.prefix_feat_cond.shape());
    assert!(
        values(&out.cond).iter().all(|v| *v == 0.0),
        "patch 0 must condition on zeros, never the reference audio"
    );

    assert_rows_close(&values(&out.mu), &mus, "mu");
    assert_rows_close(&values(&out.curr_embed), &embeds, "curr_embed");
}

/// A shape-mismatched `target_patches` is `Err`, never a panic.
#[test]
fn teacher_forced_conditioning_rejects_bad_shapes() {
    let (client, device) = cpu_setup();
    let fx = fixture(false, &device);
    let generator = fx.generator();
    let st = state(&fx, &device);

    let bad = numr::tensor::Tensor::<CpuRuntime>::zeros(
        &[2, 3, 5], // wrong patch_size/feat_dim
        st.prefill.lm_hidden.tensor().dtype(),
        &device,
    )
    .expect("zeros");

    assert!(
        generator
            .teacher_forced_conditioning(&client, &st.prefill, &bad)
            .is_err(),
        "a shape-mismatched target_patches must error, not panic"
    );

    let empty = numr::tensor::Tensor::<CpuRuntime>::zeros(
        &[0, PATCH_SIZE, FEAT_DIM],
        st.prefill.lm_hidden.tensor().dtype(),
        &device,
    )
    .expect("zeros");
    assert!(
        generator
            .teacher_forced_conditioning(&client, &st.prefill, &empty)
            .is_err(),
        "T = 0 must error, not panic"
    );
}
