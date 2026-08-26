use super::super::loudness::peak_dbfs;
use super::*;

const RATE: u32 = 48_000;

fn tone(amp: f64, secs: f64, freq: f64) -> Vec<f32> {
    let n = (RATE as f64 * secs) as usize;
    (0..n)
        .map(|i| (amp * (std::f64::consts::TAU * freq * i as f64 / RATE as f64).sin()) as f32)
        .collect()
}

#[test]
fn nothing_under_the_ceiling_is_touched() {
    let quiet = tone(0.2, 1.0, 200.0);
    let (out, r) = limit(&quiet, RATE, LimiterOptions::default());
    assert_eq!(out, quiet);
    assert_eq!(r.max_reduction_db, 0.0);
    assert_eq!(r.sustained_reduction_db, 0.0);
    assert_eq!(r.fraction_reduced, 0.0);
}

#[test]
fn the_ceiling_holds_on_a_loud_tone() {
    let loud = tone(0.95, 1.0, 200.0);
    let (out, r) = limit(&loud, RATE, LimiterOptions::default());
    let peak = peak_dbfs(&out);
    assert!(
        peak <= -1.0 + 1e-4,
        "peak {peak:.3} dBFS breached the ceiling"
    );
    assert!(r.max_reduction_db > 0.0);
}

#[test]
fn the_ceiling_holds_on_an_isolated_transient() {
    // The case gain-clamping handles badly and a limiter exists for: one spike
    // in an otherwise quiet signal. The rest of the signal must survive it.
    // Three seconds, so the ~20 ms the limiter spends on one spike is under
    // the 1% that `sustained_reduction_db` measures. That is the point of the
    // metric: brief is not sustained.
    let mut s = tone(0.1, 3.0, 200.0);
    let spike = RATE as usize * 3 / 2;
    s[spike] = 0.99;
    s[spike + 1] = -0.99;

    let (out, r) = limit(&s, RATE, LimiterOptions::default());
    let peak = peak_dbfs(&out);
    assert!(peak <= -1.0 + 1e-4, "peak {peak:.3} dBFS");

    // Far from the spike the signal is untouched.
    let far = RATE as usize / 4;
    assert!(
        (out[far] - s[far]).abs() < 1e-6,
        "a spike at 0.5 s changed the signal at {far}"
    );
    assert!(
        r.fraction_reduced < 0.05,
        "one spike ducked {:.1}% of the file",
        r.fraction_reduced * 100.0
    );
    // A lone transient is not pumping, however deeply it was reduced.
    assert!(r.max_reduction_db > 0.5, "{:.2} dB", r.max_reduction_db);
    assert_eq!(
        r.sustained_reduction_db, 0.0,
        "one spike registered as {:.2} dB of sustained reduction",
        r.sustained_reduction_db
    );
}

#[test]
fn continuous_limiting_registers_as_sustained() {
    // The other half of the contract above: a signal over the ceiling
    // throughout must drive the sustained figure, since that is pumping.
    let loud = tone(0.95, 1.0, 200.0);
    let (_, r) = limit(&loud, RATE, LimiterOptions::default());
    assert!(
        r.sustained_reduction_db > 0.3,
        "a tone 0.6 dB over the ceiling gave {:.2} dB sustained",
        r.sustained_reduction_db
    );
    assert!(r.fraction_reduced > 0.5, "{:.2}", r.fraction_reduced);
}

#[test]
fn the_envelope_never_overshoots_anywhere() {
    // The safety property stated in the module doc, checked sample by sample
    // on a signal whose amplitude sweeps through the ceiling repeatedly.
    let n = RATE as usize * 2;
    let s: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            let env = 0.5 + 0.5 * (std::f64::consts::TAU * 3.0 * t).sin();
            (env * 1.2 * (std::f64::consts::TAU * 220.0 * t).sin()) as f32
        })
        .collect();
    let (out, _) = limit(&s, RATE, LimiterOptions::default());
    let ceiling = 10f64.powf(-1.0 / 20.0) as f32;
    for (i, &x) in out.iter().enumerate() {
        assert!(
            x.abs() <= ceiling + 1e-5,
            "sample {i} is {x} against a ceiling of {ceiling}"
        );
    }
}

#[test]
fn the_gain_envelope_is_smooth_enough_not_to_click() {
    // A limiter that switches gain abruptly is audible as a click. Bounding
    // the per-sample change is what the running mean buys.
    let mut s = tone(0.1, 1.0, 200.0);
    s[RATE as usize / 2] = 0.99;
    let (out, _) = limit(&s, RATE, LimiterOptions::default());

    let ratio = |i: usize| {
        if s[i].abs() < 1e-6 {
            1.0
        } else {
            (out[i] / s[i]) as f64
        }
    };
    let mut worst = 0.0f64;
    for i in 1..s.len() {
        if s[i].abs() > 1e-3 && s[i - 1].abs() > 1e-3 {
            worst = worst.max((ratio(i) - ratio(i - 1)).abs());
        }
    }
    assert!(worst < 0.01, "gain jumped {worst:.4} in one sample");
}

#[test]
fn a_wider_window_ducks_more_of_the_signal() {
    let mut s = tone(0.1, 1.0, 200.0);
    s[RATE as usize / 2] = 0.99;
    let narrow = limit(
        &s,
        RATE,
        LimiterOptions {
            window_ms: 1.0,
            ..Default::default()
        },
    )
    .1;
    let wide = limit(
        &s,
        RATE,
        LimiterOptions {
            window_ms: 20.0,
            ..Default::default()
        },
    )
    .1;
    assert!(
        wide.fraction_reduced > narrow.fraction_reduced * 4.0,
        "20 ms ducked {:.4}, 1 ms ducked {:.4}",
        wide.fraction_reduced,
        narrow.fraction_reduced
    );
}

#[test]
fn empty_input_is_handled() {
    let (out, r) = limit(&[], RATE, LimiterOptions::default());
    assert!(out.is_empty());
    assert_eq!(r.max_reduction_db, 0.0);
    assert_eq!(r.sustained_reduction_db, 0.0);
}

#[test]
fn the_running_min_matches_a_direct_scan() {
    // The deque is an optimization; this pins it against the obvious version.
    let src: Vec<f64> = (0..500).map(|i| ((i * 37) % 101) as f64 / 101.0).collect();
    for half in [0usize, 1, 7, 64, 600] {
        for (i, &got) in running_min(&src, half).iter().enumerate() {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(src.len());
            let slow = src[lo..hi].iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                (got - slow).abs() < 1e-12,
                "half={half} i={i}: {got} vs {slow}"
            );
        }
    }
}
