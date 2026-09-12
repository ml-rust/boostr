//! Shared fixtures for the enhancement-chain tests: a sample rate, a noise
//! generator, and the phrase/gap timing both `enhance_denoise` and
//! `enhance_pipeline` build their synthetic takes from.

pub const RATE: u32 = 48_000;

/// A deterministic pseudo-random hiss in `[-amp, amp]`.
///
/// A fixed LCG rather than a crate: the test must fail or pass identically on
/// every machine, and a seeded generator is the only way to assert a specific
/// dB improvement.
pub fn hiss(n: usize, amp: f32, seed: u64) -> Vec<f32> {
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

/// Phrase length and gap length, in samples. Real speech is bursts separated
/// by pauses, and the pauses are what a gate measures its floor from.
pub const PHRASE: usize = RATE as usize * 4 / 5;
pub const GAP: usize = RATE as usize * 2 / 5;

/// True where a sample falls inside a spoken phrase.
pub fn is_voiced(i: usize) -> bool {
    i % (PHRASE + GAP) < PHRASE
}
