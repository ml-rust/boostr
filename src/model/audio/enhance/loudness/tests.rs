use super::*;

const RATE: u32 = 48_000;

/// A 1 kHz sine at the given peak amplitude, `secs` long.
fn tone(amp: f64, secs: f64) -> Vec<f32> {
    let n = (RATE as f64 * secs) as usize;
    (0..n)
        .map(|i| (amp * (std::f64::consts::TAU * 1000.0 * i as f64 / RATE as f64).sin()) as f32)
        .collect()
}

#[test]
fn silence_has_no_loudness_rather_than_a_number() {
    let lufs = integrated_lufs(&vec![0.0f32; RATE as usize], RATE).unwrap();
    assert!(lufs.is_infinite() && lufs < 0.0, "got {lufs}");
}

#[test]
fn input_shorter_than_one_block_is_unmeasurable() {
    // A 400 ms block is the smallest thing BS.1770 integrates. Returning a
    // number for 100 ms would invite normalizing against nothing.
    let lufs = integrated_lufs(&tone(0.5, 0.1), RATE).unwrap();
    assert!(
        lufs.is_infinite(),
        "100 ms should not produce a reading, got {lufs}"
    );
    assert!(integrated_lufs(&tone(0.5, 1.0), RATE).unwrap().is_finite());
}

#[test]
fn doubling_amplitude_raises_loudness_by_six_db() {
    let quiet = integrated_lufs(&tone(0.25, 3.0), RATE).unwrap();
    let loud = integrated_lufs(&tone(0.5, 3.0), RATE).unwrap();
    assert!(
        (loud - quiet - 6.02).abs() < 0.05,
        "{quiet:.2} -> {loud:.2} should be +6.02 LU"
    );
}

#[test]
fn normalizing_hits_the_target() {
    let out = normalize_to_lufs(&tone(0.1, 3.0), RATE, -23.0, -1.0).unwrap();
    let got = integrated_lufs(&out, RATE).unwrap();
    assert!((got - -23.0).abs() < 0.1, "got {got:.2} LUFS, wanted -23");
}

#[test]
fn the_peak_ceiling_overrides_the_loudness_target() {
    // A very quiet source needs a large gain to reach -16 LUFS; the ceiling
    // must stop it short rather than clip. Clipping a reference destroys it for
    // voice cloning no matter how correct the loudness is.
    let out = normalize_to_lufs(&tone(0.001, 3.0), RATE, -16.0, -1.0).unwrap();
    let peak = peak_dbfs(&out);
    assert!(
        peak <= -1.0 + 1e-6,
        "peak {peak:.2} dBFS breached the ceiling"
    );
    let lufs = integrated_lufs(&out, RATE).unwrap();
    assert!(lufs < -16.0, "must fall short of target, got {lufs:.2}");
}

#[test]
fn the_gate_ignores_silence_between_words() {
    // The point of gating: a signal with long gaps must measure the same as the
    // speech alone. RMS over the whole thing would read far quieter.
    let speech = tone(0.5, 2.0);
    let mut gapped = speech.clone();
    gapped.extend(vec![0.0f32; RATE as usize * 4]);
    gapped.extend(tone(0.5, 2.0));

    let solid = integrated_lufs(&speech, RATE).unwrap();
    let with_gaps = integrated_lufs(&gapped, RATE).unwrap();
    assert!(
        (solid - with_gaps).abs() < 0.5,
        "gated loudness {with_gaps:.2} should match ungapped {solid:.2}"
    );

    // Ungated RMS does NOT have this property, which is why it is not used.
    let rms_db = |s: &[f32]| {
        20.0 * (s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / s.len() as f64)
            .sqrt()
            .log10()
    };
    assert!(
        rms_db(&speech) - rms_db(&gapped) > 2.0,
        "RMS should be badly skewed by the gaps"
    );
}

#[test]
fn a_zero_sample_rate_is_rejected() {
    assert!(integrated_lufs(&tone(0.5, 1.0), 0).is_err());
}
