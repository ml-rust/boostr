use super::*;

const RATE: u32 = 48_000;

/// A deterministic pseudo-random hiss in `[-amp, amp]`.
///
/// A fixed LCG rather than a crate: the test must fail or pass identically on
/// every machine, and a seeded generator is the only way to assert a specific
/// dB improvement.
fn hiss(n: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            (u as f32) * amp
        })
        .collect()
}

/// A voiced-speech stand-in: a 150 Hz fundamental with four harmonics.
fn voiced(i: usize, amp: f32) -> f32 {
    let t = i as f64 / RATE as f64;
    let mut v = 0.0;
    for h in 1..=5 {
        v += (1.0 / h as f64) * (std::f64::consts::TAU * 150.0 * h as f64 * t).sin();
    }
    (v * amp as f64 / 2.28) as f32
}

/// Phrase length and gap length, in samples. Real speech is bursts separated
/// by pauses, and the pauses are what a gate measures its floor from.
const PHRASE: usize = RATE as usize * 4 / 5;
const GAP: usize = RATE as usize * 2 / 5;

/// True where a sample falls inside a spoken phrase.
fn is_voiced(i: usize) -> bool {
    i % (PHRASE + GAP) < PHRASE
}

/// A take: phrases separated by pauses, with hiss throughout.
fn take(n: usize, voice_amp: f32, noise_amp: f32, seed: u64) -> Vec<f32> {
    let noise = hiss(n, noise_amp, seed);
    (0..n)
        .map(|i| {
            noise[i]
                + if is_voiced(i) {
                    voiced(i, voice_amp)
                } else {
                    0.0
                }
        })
        .collect()
}

fn rms_db(s: &[f32]) -> f64 {
    let p = s.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / s.len() as f64;
    if p > 0.0 {
        10.0 * p.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// RMS over only the samples the predicate selects, skipping `pad` samples on
/// either side of every boundary so frames straddling a transition are excluded.
fn rms_db_where(s: &[f32], pad: usize, want_voiced: bool) -> f64 {
    let sel: Vec<f32> = (pad..s.len() - pad)
        .filter(|&i| {
            is_voiced(i) == want_voiced
                && is_voiced(i - pad) == want_voiced
                && is_voiced(i + pad) == want_voiced
        })
        .map(|i| s[i])
        .collect();
    rms_db(&sel)
}

/// Energy at one frequency, as amplitude. Used to check the voice survives in
/// the band it actually occupies rather than only in aggregate.
fn energy_at(s: &[f32], f: f64) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &x) in s.iter().enumerate() {
        let th = std::f64::consts::TAU * f * i as f64 / RATE as f64;
        re += x as f64 * th.cos();
        im += x as f64 * th.sin();
    }
    (re * re + im * im).sqrt() / s.len() as f64
}

#[test]
fn the_pauses_go_quiet_and_the_phrases_do_not() {
    // The whole point of the module in one assertion pair.
    let input = take(RATE as usize * 6, 0.3, 0.01, 7);
    let out = denoise(&input, DenoiseOptions::default()).unwrap();
    assert_eq!(out.len(), input.len());

    let pad = 2048;
    let quiet_in = rms_db_where(&input, pad, false);
    let quiet_out = rms_db_where(&out, pad, false);
    let loud_in = rms_db_where(&input, pad, true);
    let loud_out = rms_db_where(&out, pad, true);

    assert!(
        quiet_in - quiet_out > 12.0,
        "pauses fell only {:.1} dB ({quiet_in:.1} -> {quiet_out:.1})",
        quiet_in - quiet_out
    );
    assert!(
        loud_in - loud_out < 3.0,
        "phrases lost {:.1} dB ({loud_in:.1} -> {loud_out:.1})",
        loud_in - loud_out
    );
}

#[test]
fn the_voice_keeps_its_fundamental() {
    // Aggregate RMS can stay put while the gate eats the harmonics and leaves
    // broadband residue. This checks the band the voice actually lives in.
    // An earlier noise estimator — a per-bin percentile over the whole take —
    // passed the RMS check above and still cost 28 dB of fundamental here,
    // because a sustained vowel is stationary in its own bins.
    let input = take(RATE as usize * 6, 0.3, 0.01, 7);
    let out = denoise(&input, DenoiseOptions::default()).unwrap();
    let before = energy_at(&input, 150.0);
    let after = energy_at(&out, 150.0);
    assert!(
        20.0 * (after / before).log10() > -2.0,
        "150 Hz fundamental fell {:.1} dB",
        20.0 * (after / before).log10()
    );
}

#[test]
fn a_clean_recording_is_left_almost_alone() {
    // Guards the opposite failure: a gate that always attenuates would pass
    // every test above by suppressing everything.
    let input = take(RATE as usize * 6, 0.3, 0.0, 7);
    let out = denoise(&input, DenoiseOptions::default()).unwrap();
    let pad = 2048;
    let before = rms_db_where(&input, pad, true);
    let after = rms_db_where(&out, pad, true);
    assert!(
        (before - after).abs() < 1.5,
        "clean phrases moved {:.2} dB ({before:.1} -> {after:.1})",
        before - after
    );
}

#[test]
fn a_take_that_is_mostly_silence_is_still_gated() {
    // One sentence with the recorder left running, which is what a reference
    // take usually is. Here the MEDIAN frame is the noise floor, so a guard
    // that compares the quiet frames against the median reads this as
    // pause-free and refuses to gate. A real 30 s take measured 1.9 dB of
    // range against the median and 24.2 dB against the 90th percentile.
    let n = RATE as usize * 10;
    let noise = hiss(n, 0.01, 31);
    let speech_start = RATE as usize * 4;
    let speech_end = RATE as usize * 5;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            noise[i]
                + if (speech_start..speech_end).contains(&i) {
                    voiced(i, 0.3)
                } else {
                    0.0
                }
        })
        .collect();

    let out = denoise(&input, DenoiseOptions::default()).unwrap();
    let pad = 4096;
    let silent_in = rms_db(&input[pad..speech_start - pad]);
    let silent_out = rms_db(&out[pad..speech_start - pad]);
    assert!(
        silent_in - silent_out > 12.0,
        "silence fell only {:.1} dB ({silent_in:.1} -> {silent_out:.1})",
        silent_in - silent_out
    );
    let spoken_in = rms_db(&input[speech_start + pad..speech_end - pad]);
    let spoken_out = rms_db(&out[speech_start + pad..speech_end - pad]);
    assert!(
        spoken_in - spoken_out < 3.0,
        "the one sentence lost {:.1} dB",
        spoken_in - spoken_out
    );
}

#[test]
fn a_take_with_no_pause_is_left_untouched() {
    // Pure hiss, or an unbroken sustained note: no frame is quieter than any
    // other, so no floor can be measured. Gating anyway would subtract the
    // signal from itself.
    let input = hiss(RATE as usize * 4, 0.01, 11);
    let out = denoise(&input, DenoiseOptions::default()).unwrap();
    assert_eq!(out, input);

    let unbroken: Vec<f32> = (0..RATE as usize * 4).map(|i| voiced(i, 0.3)).collect();
    assert_eq!(
        denoise(&unbroken, DenoiseOptions::default()).unwrap(),
        unbroken
    );
}

#[test]
fn an_explicit_noise_clip_gates_a_take_that_has_no_pause() {
    // The reason `denoise_with_profile` exists: room tone captured separately
    // lets a pause-free take be cleaned, which the automatic path refuses.
    //
    // Also the guard on `freq_smooth_bins` defaulting to 0. Restoring the
    // old default of 2 turns this test's 15 dB win into a 9 dB loss.
    let n = RATE as usize * 4;
    let noise = hiss(n * 2, 0.01, 23);
    let unbroken: Vec<f32> = (0..n).map(|i| voiced(i, 0.3) + noise[n + i]).collect();

    let out = denoise_with_profile(&unbroken, &noise[..n], DenoiseOptions::default()).unwrap();
    assert_ne!(out, unbroken);

    let pad = 4096;
    let clean: Vec<f32> = (0..n).map(|i| voiced(i, 0.3)).collect();
    let err_before = rms_db(
        &(pad..n - pad)
            .map(|i| unbroken[i] - clean[i])
            .collect::<Vec<_>>(),
    );
    let err_after = rms_db(
        &(pad..n - pad)
            .map(|i| out[i] - clean[i])
            .collect::<Vec<_>>(),
    );
    assert!(
        err_before - err_after > 6.0,
        "error against the clean signal fell only {:.1} dB ({err_before:.1} -> {err_after:.1})",
        err_before - err_after
    );
}

#[test]
fn the_gain_floor_bounds_how_far_anything_is_pushed_down() {
    // A cell is attenuated, never zeroed: isolated survivors next to digital
    // silence are what "musical noise" sounds like. Checked on the mask
    // directly, because aggregate RMS over a region mixes cells at the floor
    // with cells that passed.
    let input = take(RATE as usize * 6, 0.3, 0.01, 7);
    let deep = DenoiseOptions {
        gain_floor_db: -20.0,
        ..Default::default()
    };
    let out = denoise(&input, deep).unwrap();
    let pad = 2048;
    let drop = rms_db_where(&input, pad, false) - rms_db_where(&out, pad, false);
    assert!(
        drop <= 20.0 + 1e-6,
        "pauses fell {drop:.1} dB, past the 20 dB floor"
    );

    // And the floor is a real control: a deeper one suppresses further.
    let deeper = DenoiseOptions {
        gain_floor_db: -40.0,
        ..Default::default()
    };
    let out2 = denoise(&input, deeper).unwrap();
    let drop2 = rms_db_where(&input, pad, false) - rms_db_where(&out2, pad, false);
    assert!(
        drop2 > drop + 2.0,
        "a 40 dB floor gave {drop2:.1} dB where a 20 dB floor gave {drop:.1} dB"
    );
}

#[test]
fn input_shorter_than_one_frame_comes_back_untouched() {
    let short: Vec<f32> = (0..256).map(|i| voiced(i, 0.3)).collect();
    let out = denoise(&short, DenoiseOptions::default()).unwrap();
    assert_eq!(out, short);
}

#[test]
fn bad_options_are_rejected() {
    let v = take(RATE as usize, 0.3, 0.01, 7);
    assert!(
        denoise(
            &v,
            DenoiseOptions {
                n_fft: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        denoise(
            &v,
            DenoiseOptions {
                noise_frame_fraction: 0.0,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn a_noise_clip_shorter_than_one_frame_is_rejected() {
    let v = take(RATE as usize, 0.3, 0.01, 7);
    assert!(denoise_with_profile(&v, &v[..512], DenoiseOptions::default()).is_err());
}

#[test]
fn the_noise_floor_estimate_tracks_the_actual_floor() {
    let quiet = noise_floor_dbfs(&hiss(RATE as usize, 0.001, 3), RATE);
    let loud = noise_floor_dbfs(&hiss(RATE as usize, 0.01, 3), RATE);
    assert!(
        (loud - quiet - 20.0).abs() < 1.0,
        "10x amplitude should read +20 dB, got {:.2}",
        loud - quiet
    );
}
