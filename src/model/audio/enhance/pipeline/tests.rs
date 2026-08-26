use super::super::denoise::DenoiseOptions;
use super::*;

const RATE: u32 = 48_000;

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

const PHRASE: usize = RATE as usize * 4 / 5;
const GAP: usize = RATE as usize * 2 / 5;

fn is_voiced(i: usize) -> bool {
    i % (PHRASE + GAP) < PHRASE
}

/// Everything the chain contributes with nobody speaking: hiss plus 25 Hz
/// desk rumble, inaudible but level-hogging.
fn room_tone(n: usize, seed: u64) -> Vec<f32> {
    let noise = hiss(n, 0.004, seed);
    (0..n)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            noise[i] + (0.05 * (std::f64::consts::TAU * 25.0 * t).sin()) as f32
        })
        .collect()
}

/// A quiet, noisy, rumbling take with pauses in it — what an untreated
/// recording of someone actually speaking looks like.
fn raw_take(n: usize) -> Vec<f32> {
    let room = room_tone(n, 19);
    (0..n)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            let mut v = 0.0;
            if is_voiced(i) {
                for h in 1..=5 {
                    v += (1.0 / h as f64) * (std::f64::consts::TAU * 150.0 * h as f64 * t).sin();
                }
            }
            room[i] + (v * 0.04 / 2.28) as f32
        })
        .collect()
}

/// Energy at one frequency, as amplitude.
fn energy_at(s: &[f32], f: f64) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &x) in s.iter().enumerate() {
        let th = std::f64::consts::TAU * f * i as f64 / RATE as f64;
        re += x as f64 * th.cos();
        im += x as f64 * th.sin();
    }
    ((re * re + im * im).sqrt() / s.len() as f64).max(1e-12)
}

/// Level at `f` relative to the 150 Hz fundamental, in dB. Ratios rather than
/// absolutes, because the loudness stage rescales everything at the end and an
/// absolute level says nothing about what the tone stages did.
fn relative_db(s: &[f32], f: f64) -> f64 {
    20.0 * (energy_at(s, f) / energy_at(s, 150.0)).log10()
}

/// A take shaped like real speech: quiet overall, with peaks 20 dB above its
/// loudness. Gain alone cannot bring this to a target under a ceiling.
fn peaky_take(n: usize) -> Vec<f32> {
    let mut s = raw_take(n);
    // One plosive per phrase, where a real one lands.
    let mut i = 0;
    while i < n {
        if is_voiced(i) && i % (PHRASE + GAP) < 600 {
            let k = (i % (PHRASE + GAP)) as f64 / 600.0;
            s[i] += (0.6 * (1.0 - k) * (std::f64::consts::TAU * 120.0 * k).sin()) as f32;
        }
        i += 1;
    }
    s
}

#[test]
fn a_peaky_take_still_reaches_the_loudness_target() {
    // Gain alone misses by the peak-to-loudness ratio. A measured 30 s
    // reference take was -38 LUFS with a -15 dBFS peak: 23 dB apart, so a
    // -18 LUFS target under a -1 dBFS ceiling is 19 dB out of reach without
    // the limiter.
    let input = peaky_take(RATE as usize * 6);
    let opts = EnhanceOptions::default();

    let (_, with) = enhance(&input, RATE, opts).unwrap();
    let (_, without) = enhance(
        &input,
        RATE,
        EnhanceOptions {
            limiter_window_ms: None,
            ..opts
        },
    )
    .unwrap();

    assert!(
        with.output_lufs - without.output_lufs > 3.0,
        "the limiter bought only {:.2} LU ({:.2} -> {:.2})",
        with.output_lufs - without.output_lufs,
        without.output_lufs,
        with.output_lufs
    );
    assert!(
        !without.reached_target,
        "the fixture is not peaky enough to need a limiter: gain alone reached {:.2} LUFS",
        without.output_lufs
    );
    // The default 6 dB cap gets close but need not arrive; a cap wide enough
    // to spend does.
    let (_, generous) = enhance(
        &input,
        RATE,
        EnhanceOptions {
            max_limiting_db: 12.0,
            ..opts
        },
    )
    .unwrap();
    assert!(
        generous.reached_target,
        "with 12 dB of limiting allowed, landed at {:.2} LUFS",
        generous.output_lufs
    );
    assert!(with.limiter_reduction_db > 0.0);
    assert_eq!(without.limiter_reduction_db, 0.0);
}

#[test]
fn the_limiting_cap_is_honoured_even_at_the_cost_of_the_target() {
    // A reference that pumps teaches the pumping to the model, so the cap
    // outranks the loudness target.
    let input = peaky_take(RATE as usize * 6);
    let (out, r) = enhance(
        &input,
        RATE,
        EnhanceOptions {
            target_lufs: -6.0,
            max_limiting_db: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        r.limiter_reduction_db <= 2.0 + 1e-6,
        "limiter applied {:.2} dB sustained against a 2 dB cap",
        r.limiter_reduction_db
    );
    assert!(!r.reached_target, "reached {:.2} LUFS", r.output_lufs);
    assert!(r.output_peak_dbfs <= -1.0 + 1e-4);
    assert!(out.iter().all(|s| s.is_finite()));
}

#[test]
fn a_raw_take_comes_out_at_the_target_and_under_the_ceiling() {
    let (out, report) = enhance(
        &raw_take(RATE as usize * 6),
        RATE,
        EnhanceOptions::default(),
    )
    .unwrap();
    assert!(
        (report.output_lufs - -18.0).abs() < 1.0,
        "landed at {:.2} LUFS",
        report.output_lufs
    );
    assert!(
        report.output_peak_dbfs <= -1.0 + 1e-6,
        "peak {:.2} dBFS breached the ceiling",
        report.output_peak_dbfs
    );
    assert_eq!(out.len(), RATE as usize * 6);
    assert!(out.iter().all(|s| s.is_finite()));
}

#[test]
fn the_chain_raises_the_signal_further_above_its_noise_floor() {
    // The measurement that says the chain did its job: loudness minus noise
    // floor. Gain alone cannot improve it, because gain lifts both.
    let input = raw_take(RATE as usize * 6);
    let (_, report) = enhance(&input, RATE, EnhanceOptions::default()).unwrap();
    let before = report.input_lufs - report.input_noise_floor_dbfs;
    let after = report.output_lufs - report.output_noise_floor_dbfs;
    assert!(
        after - before > 6.0,
        "signal-to-floor improved only {:.1} dB ({before:.1} -> {after:.1})",
        after - before
    );
}

#[test]
fn a_supplied_noise_clip_is_never_worse_than_hunting_for_pauses() {
    // Why `enhance_with_noise_profile` is worth offering. Pause-hunting takes
    // the quietest tenth of frames, and in a real take some of those still
    // catch the tail of a word, which inflates the profile. Dedicated room
    // tone does not. Measured here at 24.3 dB for pause-hunting against
    // 35.2 dB for the supplied clip; the assertion below only pins the
    // direction, since the exact margin depends on the take.
    let input = raw_take(RATE as usize * 6);
    let room = room_tone(RATE as usize * 2, 41);

    let auto = enhance(&input, RATE, EnhanceOptions::default()).unwrap().1;
    let given = enhance_with_noise_profile(&input, Some(&room), RATE, EnhanceOptions::default())
        .unwrap()
        .1;

    let auto_snr = auto.output_lufs - auto.output_noise_floor_dbfs;
    let given_snr = given.output_lufs - given.output_noise_floor_dbfs;
    assert!(
        given_snr >= auto_snr - 0.5,
        "supplied profile gave {given_snr:.1} dB, pause-hunting gave {auto_snr:.1} dB"
    );
}

#[test]
fn the_rumble_is_gone() {
    let input = raw_take(RATE as usize * 6);
    let (out, _) = enhance(&input, RATE, EnhanceOptions::default()).unwrap();
    let before = relative_db(&input, 25.0);
    let after = relative_db(&out, 25.0);
    // The 70 Hz two-pole high-pass is 18 dB down at 25 Hz; the bass shelf then
    // hands about 1.7 dB of that back. Anything much short of 14 dB means the
    // filter did not run.
    assert!(
        before - after > 14.0,
        "rumble relative to voice fell only {:.1} dB ({before:.1} -> {after:.1})",
        before - after
    );
}

#[test]
fn denoising_can_be_turned_off_without_changing_the_rest() {
    let input = raw_take(RATE as usize * 6);
    let (_, report) = enhance(
        &input,
        RATE,
        EnhanceOptions {
            denoise: None,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((report.output_lufs - -18.0).abs() < 1.0);

    // The high-pass alone already improves signal-to-floor, because rumble is
    // most of what the floor measurement was seeing. The claim to check is
    // narrower: the gate is responsible for a further, large share of it.
    let gated = enhance(&input, RATE, EnhanceOptions::default()).unwrap().1;
    let off = report.output_lufs - report.output_noise_floor_dbfs;
    let on = gated.output_lufs - gated.output_noise_floor_dbfs;
    assert!(
        on - off > 6.0,
        "the gate added only {:.1} dB over the filters alone ({off:.1} -> {on:.1})",
        on - off
    );
}

#[test]
fn empty_input_and_zero_rate_are_rejected() {
    assert!(enhance(&[], RATE, EnhanceOptions::default()).is_err());
    assert!(enhance(&raw_take(1024), 0, EnhanceOptions::default()).is_err());
}

#[test]
fn the_bass_shelf_is_a_real_control() {
    // The options are controls, not decoration. Measured as low-band relative
    // to the fundamental: the loudness stage rescales the whole signal, so an
    // absolute low-band level would mostly report that rescaling.
    let input = raw_take(RATE as usize * 6);
    let with = EnhanceOptions {
        bass_boost_db: 9.0,
        denoise: Some(DenoiseOptions::default()),
        ..Default::default()
    };
    let without = EnhanceOptions {
        bass_boost_db: 0.0,
        ..with
    };
    let (a, _) = enhance(&input, RATE, with).unwrap();
    let (b, _) = enhance(&input, RATE, without).unwrap();
    // 75 Hz sits well inside the 140 Hz shelf, 150 Hz is the reference.
    let lifted = relative_db(&a, 75.0) - relative_db(&b, 75.0);
    assert!(
        lifted > 3.0,
        "9 dB of shelf lifted the low band only {lifted:.2} dB"
    );
}

#[test]
fn the_presence_shelf_is_a_real_control() {
    let input = raw_take(RATE as usize * 6);
    let with = EnhanceOptions {
        presence_db: 6.0,
        ..Default::default()
    };
    let without = EnhanceOptions {
        presence_db: 0.0,
        ..Default::default()
    };
    let (a, _) = enhance(&input, RATE, with).unwrap();
    let (b, _) = enhance(&input, RATE, without).unwrap();
    let lifted = relative_db(&a, 12000.0) - relative_db(&b, 12000.0);
    assert!(
        lifted > 3.0,
        "6 dB of high shelf lifted the top only {lifted:.2} dB"
    );
}

#[test]
fn the_output_length_always_equals_the_input_length() {
    // Load-bearing for corpus preparation: `CorpusOptions::enhance` runs the
    // chain over a whole recording BEFORE the VAD, so every segment index the
    // VAD then produces — and every `start_sample` in the manifest — is an
    // index into a buffer of this length. A chain that returned a different
    // count would silently shift every utterance boundary.
    //
    // Lengths deliberately off any hop or FFT multiple, which is where the
    // STFT round trip would leak a discrepancy.
    for n in [4001usize, 48_000, 50_003, 96_001] {
        let (out, _) = enhance(&raw_take(n), RATE, EnhanceOptions::default()).unwrap();
        assert_eq!(out.len(), n, "{n} samples in, {} out", out.len());
    }
}
