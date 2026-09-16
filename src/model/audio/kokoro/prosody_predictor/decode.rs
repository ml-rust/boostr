/// Turn per-phoneme duration logits into an integer duration per phoneme.
///
/// * `logits [T, max_dur]` — 2-D (drop the batch axis first if needed).
/// * `min_frames` — floor for each phoneme, clamped to at least 1.
///
/// Uses the softmax-weighted expected value across duration bins, rounded and
/// clamped to `[min_frames, max_dur]`. This matches the reference Kokoro
/// implementation's `torch.sigmoid(duration).sum(axis=-1)` convention when inputs are treated as
/// per-bin probabilities — but here we stay loyal to the classification head's
/// softmax output. Callers can swap in their own decoding if the reference
/// implementation's behavior diverges.
pub fn decode_prosody_durations(
    logits: &[f32],
    t: usize,
    max_dur: usize,
    min_frames: u32,
) -> Vec<u32> {
    assert_eq!(logits.len(), t * max_dur, "logits must be [T, max_dur]");
    let floor = min_frames.max(1);
    let mut out = Vec::with_capacity(t);
    for row in 0..t {
        let base = row * max_dur;
        // softmax — for-loop, stable (subtract max).
        let mut m = f32::NEG_INFINITY;
        for d in 0..max_dur {
            m = m.max(logits[base + d]);
        }
        let mut sum_exp = 0.0f64;
        for d in 0..max_dur {
            sum_exp += ((logits[base + d] - m) as f64).exp();
        }
        // expected value under softmax
        let mut expected = 0.0f64;
        for d in 0..max_dur {
            let p = ((logits[base + d] - m) as f64).exp() / sum_exp;
            expected += p * (d as f64);
        }
        let frames = expected.round() as i64;
        let frames = frames.clamp(floor as i64, max_dur as i64 - 1).max(1);
        out.push(frames as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_prosody_durations_clamps_to_min_and_rounds() {
        // Uniform logits → expected value = (max_dur - 1)/2 = 1.0 for max_dur=3.
        let logits = vec![0.0f32; 2 * 3];
        let out = decode_prosody_durations(&logits, 2, 3, 1);
        assert_eq!(out, vec![1, 1]);
    }

    #[test]
    fn decode_prosody_durations_picks_peak() {
        // Strong logit on bin 2 → expected ≈ 2.
        let logits = vec![
            -10.0, -10.0, 10.0, // row 0 → dur 2
            -10.0, 10.0, -10.0, // row 1 → dur 1
        ];
        let out = decode_prosody_durations(&logits, 2, 3, 1);
        assert_eq!(out, vec![2, 1]);
    }
}
