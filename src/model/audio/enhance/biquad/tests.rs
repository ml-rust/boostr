use super::*;

const RATE: f64 = 48_000.0;

#[test]
fn a_highpass_cuts_below_its_corner_and_passes_above() {
    let hp = Biquad::highpass(RATE, 80.0, std::f64::consts::FRAC_1_SQRT_2);
    // Butterworth is -3 dB at the corner: 10^(-3/20) = 0.708.
    let at_corner = hp.magnitude_at(RATE, 80.0);
    assert!(
        (at_corner - 0.7079).abs() < 0.01,
        "corner gain {at_corner:.4} should be -3 dB"
    );
    assert!(hp.magnitude_at(RATE, 20.0) < 0.1, "20 Hz must be well down");
    assert!(
        hp.magnitude_at(RATE, 1000.0) > 0.99,
        "passband must be flat"
    );
}

#[test]
fn shelves_apply_their_gain_on_the_correct_side() {
    let low = Biquad::low_shelf(RATE, 200.0, 0.707, 6.0);
    // +6 dB is a linear gain of 2.
    assert!(
        (low.magnitude_at(RATE, 20.0) - 2.0).abs() < 0.05,
        "low shelf lifts lows"
    );
    assert!(
        (low.magnitude_at(RATE, 8000.0) - 1.0).abs() < 0.05,
        "and leaves highs"
    );

    let high = Biquad::high_shelf(RATE, 2000.0, 0.707, 6.0);
    assert!(
        (high.magnitude_at(RATE, 16000.0) - 2.0).abs() < 0.05,
        "high shelf lifts highs"
    );
    assert!(
        (high.magnitude_at(RATE, 100.0) - 1.0).abs() < 0.05,
        "and leaves lows"
    );
}

#[test]
fn a_shelf_at_zero_gain_is_a_pass_through() {
    let flat = Biquad::low_shelf(RATE, 200.0, 0.707, 0.0);
    for f in [20.0, 100.0, 1000.0, 10000.0] {
        let g = flat.magnitude_at(RATE, f);
        assert!((g - 1.0).abs() < 1e-9, "{f} Hz gain {g}");
    }
}

#[test]
fn filtering_a_tone_matches_the_computed_magnitude() {
    // The response function and the actual filter must agree, or tests that
    // use `magnitude_at` prove nothing about what `process` does.
    let freq = 1000.0;
    let mut hp = Biquad::highpass(RATE, 300.0, 0.707);
    let n = 48_000;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| (std::f64::consts::TAU * freq * i as f64 / RATE).sin() as f32)
        .collect();
    hp.process_buffer(&mut buf);

    // Skip the transient; measure amplitude over the steady-state tail.
    let tail = &buf[n / 2..];
    let peak = tail.iter().fold(0.0f32, |a, &b| a.max(b.abs())) as f64;
    let expected = hp.magnitude_at(RATE, freq);
    assert!(
        (peak - expected).abs() < 0.02,
        "measured {peak:.4} vs computed {expected:.4}"
    );
}

#[test]
fn reset_clears_the_tail_of_a_previous_signal() {
    let mut hp = Biquad::highpass(RATE, 100.0, 0.707);
    let mut loud = vec![1.0f32; 1000];
    hp.process_buffer(&mut loud);

    // Without reset the decaying state leaks into the next buffer as a click.
    let mut dirty = hp;
    let mut quiet_dirty = vec![0.0f32; 100];
    dirty.process_buffer(&mut quiet_dirty);
    let leak = quiet_dirty.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(leak > 1e-6, "state should leak without reset, got {leak}");

    hp.reset();
    let mut quiet_clean = vec![0.0f32; 100];
    hp.process_buffer(&mut quiet_clean);
    assert!(
        quiet_clean.iter().all(|&s| s == 0.0),
        "reset must silence it"
    );
}
