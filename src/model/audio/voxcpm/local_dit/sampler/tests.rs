//! Tests for the CFM sampler.
//!
//! The failure mode here is plausible-but-wrong output, so each test pins a
//! value the reference fixes exactly: the realized schedule, the untouched
//! warmup step, the `optimized_scale` quotient (epsilon placement included),
//! and the `cfg_value = 1.0` collapse.
//!
//! The estimator itself is the tiny synthetic one from
//! [`crate::model::audio::voxcpm::local_dit::tests`].

use super::{CfmOptions, cfg_combine, cfm_time_span, optimized_scale};
use crate::model::audio::voxcpm::local_dit::tests as fixture;
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn var(data: &[f32], shape: &[usize], device: &CpuDevice) -> Var<CpuRuntime> {
    Var::new(
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
        false,
    )
}

fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
    v.tensor().contiguous().unwrap().to_vec()
}

/// The 11 values `torch.linspace(1, 0, 11)` plus the `coef = 1.0` sway
/// actually realizes. The schedule must reproduce them BIT for bit, because
/// every later `t` and `dt` is derived from them.
///
/// Written as the shortest round-tripping `f32` literals; the reference
/// fixture prints them at full width as:
///
/// ```text
/// 1.0, 0.956434428691864, 0.9090169668197632, 0.8539904952049255,
/// 0.7877852320671082, 0.7071067690849304, 0.609017014503479,
/// 0.4910065531730652, 0.3510565459728241, 0.18768836557865143, 0.0
/// ```
const SCHEDULE_10: [f32; 11] = [
    1.0, 0.9564344, 0.90901697, 0.8539905, 0.78778523, 0.70710677, 0.609017, 0.49100655,
    0.35105655, 0.18768837, 0.0,
];

#[test]
fn time_span_reproduces_the_reference_schedule() {
    let span = cfm_time_span(10, 1.0).unwrap();
    assert_eq!(span.len(), 11);
    for (i, (got, want)) in span.iter().zip(SCHEDULE_10.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "t_span[{i}]: got {got:?}, want {want:?}"
        );
    }
}

/// `coef = 0` leaves the bare `linspace`, so the sway term is genuinely the
/// thing the previous test is measuring.
#[test]
fn zero_sway_coefficient_leaves_a_plain_linspace() {
    let span = cfm_time_span(10, 0.0).unwrap();
    for (i, got) in span.iter().enumerate() {
        let want = 1.0f32 - i as f32 / 10.0;
        assert!(
            (got - want).abs() < 1e-7,
            "t_span[{i}]: got {got:?}, want {want:?}"
        );
    }
    assert_ne!(span[1].to_bits(), SCHEDULE_10[1].to_bits());
}

#[test]
fn time_span_rejects_zero_timesteps() {
    assert!(cfm_time_span(0, 1.0).is_err());
}

struct Setup {
    z: Var<CpuRuntime>,
    mu: Var<CpuRuntime>,
    cond: Var<CpuRuntime>,
}

fn setup(batch: usize, device: &CpuDevice) -> Setup {
    Setup {
        z: Var::new(
            fixture::t(
                &[batch, fixture::FEAT_DIM, fixture::PATCH_SIZE],
                0.9,
                device,
            ),
            false,
        ),
        mu: Var::new(
            fixture::t(
                &[batch, fixture::MU_TOKENS * fixture::HIDDEN_DIM],
                1.3,
                device,
            ),
            false,
        ),
        cond: Var::new(
            fixture::t(
                &[batch, fixture::FEAT_DIM, fixture::PATCH_SIZE],
                1.7,
                device,
            ),
            false,
        ),
    }
}

/// `zero_init_steps` is 1 for an 11-entry schedule, so step 1 has zero
/// velocity and NO estimator call: `x` after it is bitwise `z`. Step 2 must
/// then move `x`, otherwise the loop is inert and the test proves nothing.
#[test]
fn warmup_step_leaves_x_exactly_equal_to_z() {
    let (client, device) = cpu_setup();
    let m = fixture::model(1, &device);
    let s = setup(2, &device);
    let span = cfm_time_span(10, 1.0).unwrap();

    let mut trace = Vec::new();
    let out = m
        .solve_euler(
            &client,
            &s.z,
            &span,
            &s.mu,
            &s.cond,
            2.0,
            true,
            Some(&mut trace),
        )
        .unwrap();

    assert_eq!(trace.len(), 10);
    let z = values(&s.z);
    let after_warmup = values(&trace[0]);
    for (i, (got, want)) in after_warmup.iter().zip(z.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "warmup step moved element {i}: {got:?} vs {want:?}"
        );
    }

    let after_second = values(&trace[1]);
    assert!(
        after_second
            .iter()
            .zip(z.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "step 2 must move x: {after_second:?} vs {z:?}"
    );
    assert_eq!(out.shape(), s.z.shape());
}

/// The warmup is gated on `use_cfg_zero_star`. With it off, step 1 integrates
/// like any other step and `x` moves immediately.
#[test]
fn disabling_cfg_zero_star_integrates_the_first_step() {
    let (client, device) = cpu_setup();
    let m = fixture::model(1, &device);
    let s = setup(2, &device);
    let span = cfm_time_span(10, 1.0).unwrap();

    let mut trace = Vec::new();
    m.solve_euler(
        &client,
        &s.z,
        &span,
        &s.mu,
        &s.cond,
        2.0,
        false,
        Some(&mut trace),
    )
    .unwrap();

    let z = values(&s.z);
    let first = values(&trace[0]);
    assert!(
        first
            .iter()
            .zip(z.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "step 1 must move x when the warmup is off: {first:?} vs {z:?}"
    );
}

#[test]
fn solve_euler_rejects_a_one_entry_schedule() {
    let (client, device) = cpu_setup();
    let m = fixture::model(1, &device);
    let s = setup(1, &device);
    assert!(
        m.solve_euler(&client, &s.z, &[1.0], &s.mu, &s.cond, 2.0, true, None)
            .is_err()
    );
}

/// Row 0: orthogonal velocities, so `dot = 0` and the scale is exactly 0.
/// Row 1: `pos = 3 * neg`, so the scale is exactly 3 — `1e-8` is far below
/// the `f32` ulp of `sum(neg^2) = 20`, and cannot perturb it.
/// Two rows also pin the reduction as PER ROW, not over the whole batch.
#[test]
fn optimized_scale_matches_hand_computed_values() {
    let (client, device) = cpu_setup();
    let shape = [2, 2, 2];
    let pos = var(&[1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 12.0, 0.0], &shape, &device);
    let neg = var(&[0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0], &shape, &device);

    let scale = optimized_scale(&client, &pos, &neg).unwrap();
    assert_eq!(scale.shape(), &[2, 1, 1]);
    assert_eq!(values(&scale), vec![0.0, 3.0]);
}

/// The `1e-8` is INSIDE the denominator sum. With an all-zero `neg` the
/// denominator is `1e-8`, so the scale is `0 / 1e-8 = 0`. Adding the epsilon
/// after the divide instead gives `0 / 0 = NaN` and this assertion fires.
#[test]
fn optimized_scale_epsilon_guards_a_zero_denominator() {
    let (client, device) = cpu_setup();
    let shape = [1, 2, 2];
    let pos = var(&[1.0, 2.0, 3.0, 4.0], &shape, &device);
    let neg = var(&[0.0, 0.0, 0.0, 0.0], &shape, &device);

    let scale = values(&optimized_scale(&client, &pos, &neg).unwrap());
    assert!(scale[0].is_finite(), "scale must be finite, got {scale:?}");
    assert_eq!(scale[0], 0.0);
}

/// `v = v_uncond * st + cfg * (v_cond - v_uncond * st)`.
///
/// At `cfg = 1.0` the two terms telescope and the result is `v_cond`. Every
/// value here is dyadic, so the collapse is exact and the assertion compares
/// bits — a swapped `v_cond`/`v_uncond`, or a missing `st`, breaks it.
/// The `cfg = 2.0` case pins the formula itself, which `cfg = 1.0` alone
/// cannot: it is the only weight the collapse does not hide.
#[test]
fn cfg_combine_collapses_to_the_conditional_velocity_at_one() {
    let (client, device) = cpu_setup();
    let shape = [1, 2, 2];
    let v_cond = var(&[1.0, 2.0, -3.0, 0.5], &shape, &device);
    let v_uncond = var(&[4.0, -1.0, 2.0, 8.0], &shape, &device);
    let st = var(&[0.5], &[1, 1, 1], &device);

    let at_one = values(&cfg_combine(&client, &v_cond, &v_uncond, &st, 1.0).unwrap());
    for (got, want) in at_one.iter().zip(values(&v_cond).iter()) {
        assert_eq!(got.to_bits(), want.to_bits(), "{at_one:?}");
    }

    // scaled = [2, -0.5, 1, 4]; delta = [-1, 2.5, -4, -3.5];
    // scaled + 2 * delta = [0, 4.5, -7, -3].
    let at_two = values(&cfg_combine(&client, &v_cond, &v_uncond, &st, 2.0).unwrap());
    assert_eq!(at_two, vec![0.0, 4.5, -7.0, -3.0]);
}

/// `sample` is `solve_euler` after a seeded draw: one seed reproduces a run,
/// and a different seed does not.
#[test]
fn sample_is_reproducible_for_a_seed() {
    let (client, device) = cpu_setup();
    let m = fixture::model(1, &device);
    let s = setup(2, &device);
    let options = CfmOptions {
        n_timesteps: 3,
        ..CfmOptions::default()
    };

    let a = values(&m.sample(&client, &s.mu, &s.cond, &options, 7).unwrap());
    let b = values(&m.sample(&client, &s.mu, &s.cond, &options, 7).unwrap());
    let c = values(&m.sample(&client, &s.mu, &s.cond, &options, 8).unwrap());

    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.len(), 2 * fixture::FEAT_DIM * fixture::PATCH_SIZE);
}
