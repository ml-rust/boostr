//! Integrated loudness to ITU-R BS.1770-4 (EBU R128), and normalization to it.
//!
//! Peak and RMS both mislead on speech. Peak is set by one plosive; RMS counts
//! silence between words, so a slow talker measures quieter than a fast one
//! saying the same thing at the same effort. LUFS applies a perceptual
//! weighting and GATES OUT the quiet parts, which is why broadcast and every
//! TTS corpus pipeline specifies it.
//!
//! Mono only, which is all this pipeline produces. Multi-channel BS.1770 sums
//! weighted channel powers with per-channel weights; the single-channel weight
//! is 1.0, so mono is the degenerate case and nothing here would need to change
//! beyond the sum.

use super::biquad::Biquad;
use crate::error::{Error, Result};

/// BS.1770 block length. The standard fixes this at 400 ms.
const BLOCK_MS: f64 = 400.0;
/// Blocks overlap by 75%, so a new block starts every 100 ms.
const BLOCK_OVERLAP: f64 = 0.75;
/// Absolute gate: blocks quieter than this never count.
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// Relative gate: after the absolute gate, drop blocks more than 10 LU below
/// the mean of what survived.
const RELATIVE_GATE_LU: f64 = -10.0;
/// The offset in the BS.1770 loudness formula.
const LOUDNESS_OFFSET: f64 = -0.691;

/// The two K-weighting stages, as BS.1770 defines them for 48 kHz.
///
/// The standard publishes literal coefficients rather than a parametric shape.
/// They are reproduced verbatim at 48 kHz; other rates rebuild the equivalent
/// shelf and high-pass, which is what every other implementation does and is
/// accurate to well under 0.1 LU across normal speech rates.
fn k_weighting(rate: f64) -> (Biquad, Biquad) {
    if (rate - 48_000.0).abs() < 1.0 {
        // Stage 1: high shelf, +4 dB above ~1.68 kHz (head/torso model).
        let stage1 = Biquad::from_normalized(
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            -1.69065929318241,
            0.73248077421585,
        );
        // Stage 2: high-pass at ~38 Hz (RLB curve).
        let stage2 = Biquad::from_normalized(1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621);
        (stage1, stage2)
    } else {
        (
            Biquad::high_shelf(rate, 1681.97, 0.7071752, 3.999844),
            Biquad::highpass(rate, 38.1354, 0.5003271),
        )
    }
}

/// Mean square of each gating block after K-weighting.
fn block_powers(samples: &[f32], rate: u32) -> Vec<f64> {
    let rate_f = rate as f64;
    let (mut s1, mut s2) = k_weighting(rate_f);

    let mut weighted: Vec<f32> = samples.to_vec();
    s1.process_buffer(&mut weighted);
    s2.process_buffer(&mut weighted);

    let block_len = ((BLOCK_MS / 1000.0) * rate_f).round() as usize;
    let step = ((block_len as f64) * (1.0 - BLOCK_OVERLAP))
        .round()
        .max(1.0) as usize;
    if block_len == 0 || weighted.len() < block_len {
        return Vec::new();
    }

    let mut powers = Vec::new();
    let mut start = 0usize;
    while start + block_len <= weighted.len() {
        let sum_sq: f64 = weighted[start..start + block_len]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum();
        powers.push(sum_sq / block_len as f64);
        start += step;
    }
    powers
}

/// Loudness in LUFS from a mean-square power.
fn power_to_lufs(power: f64) -> f64 {
    if power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    LOUDNESS_OFFSET + 10.0 * power.log10()
}

/// Integrated loudness in LUFS, gated per BS.1770-4.
///
/// Returns [`f64::NEG_INFINITY`] for silence, and for input shorter than one
/// 400 ms block — there is nothing to integrate, and reporting a number there
/// would invite normalizing against a measurement that does not exist.
pub fn integrated_lufs(samples: &[f32], rate: u32) -> Result<f64> {
    if rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: "sample rate is 0".to_string(),
        });
    }
    let powers = block_powers(samples, rate);
    if powers.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    // Absolute gate.
    let kept: Vec<f64> = powers
        .iter()
        .copied()
        .filter(|&p| power_to_lufs(p) > ABSOLUTE_GATE_LUFS)
        .collect();
    if kept.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    // Relative gate, referenced to the mean of what the absolute gate kept.
    let mean = kept.iter().sum::<f64>() / kept.len() as f64;
    let threshold = power_to_lufs(mean) + RELATIVE_GATE_LU;
    let final_kept: Vec<f64> = kept
        .into_iter()
        .filter(|&p| power_to_lufs(p) > threshold)
        .collect();
    if final_kept.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    let final_mean = final_kept.iter().sum::<f64>() / final_kept.len() as f64;
    Ok(power_to_lufs(final_mean))
}

/// Highest absolute sample, as dBFS. `-inf` for digital silence.
pub fn peak_dbfs(samples: &[f32]) -> f64 {
    let peak = samples.iter().fold(0.0f64, |a, &b| a.max((b as f64).abs()));
    if peak > 0.0 {
        20.0 * peak.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Scale `samples` to `target_lufs`, then pull back if that would exceed
/// `peak_ceiling_dbfs`.
///
/// The ceiling wins. Overshooting it to hit a loudness target is how a
/// normalizer produces clipping, and a clipped reference is unusable for voice
/// cloning at any loudness.
///
/// Returns the samples unchanged when the input is silent or too short to
/// measure, rather than applying an arbitrary gain to something unmeasured.
pub fn normalize_to_lufs(
    samples: &[f32],
    rate: u32,
    target_lufs: f64,
    peak_ceiling_dbfs: f64,
) -> Result<Vec<f32>> {
    let current = integrated_lufs(samples, rate)?;
    if !current.is_finite() {
        return Ok(samples.to_vec());
    }

    let mut gain = 10f64.powf((target_lufs - current) / 20.0);

    let peak = peak_dbfs(samples);
    if peak.is_finite() {
        let ceiling_gain = 10f64.powf((peak_ceiling_dbfs - peak) / 20.0);
        if ceiling_gain < gain {
            gain = ceiling_gain;
        }
    }

    Ok(samples.iter().map(|&s| (s as f64 * gain) as f32).collect())
}

#[cfg(test)]
mod tests;
