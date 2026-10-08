use crate::error::{Error, Result};

/// Turn per-phoneme duration logits into an integer duration per phoneme,
/// as reference Kokoro's `forward_with_tokens` does:
/// `round(sigmoid(logits).sum(-1) / speed).clamp(min=1)`.
///
/// * `logits [T, max_dur]` — 2-D (drop the batch axis first if needed).
/// * `speed` — `1.0` is the voice's natural pace, `2.0` twice as fast.
///
/// Each bin's sigmoid adds up to the frames a phoneme lasts. The sum is in
/// f32, rounded half to even as `torch.round` rounds, and at least 1.
/// Errors on a `logits` length other than `t * max_dur`, and on a speed
/// that is not finite and positive.
pub fn decode_prosody_durations(
    logits: &[f32],
    t: usize,
    max_dur: usize,
    speed: f32,
) -> Result<Vec<u32>> {
    if logits.len() != t * max_dur {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "expected [T={t}, max_dur={max_dur}] = {} values, got {}",
                t * max_dur,
                logits.len()
            ),
        });
    }
    if !(speed.is_finite() && speed > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "speed",
            reason: format!("expected a finite positive pace, got {speed}"),
        });
    }
    Ok(logits
        .chunks(max_dur.max(1))
        .take(t)
        .map(|row| {
            let frames: f32 = row.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).sum();
            (frames / speed).round_ties_even().max(1.0) as u32
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_bin_adds_its_sigmoid() {
        // Two bins near 1 and one near 0: two frames. All near 0: one.
        let logits = vec![20.0, 20.0, -20.0, -20.0, -20.0, -20.0];
        assert_eq!(
            decode_prosody_durations(&logits, 2, 3, 1.0).ok(),
            Some(vec![2, 1])
        );
    }

    #[test]
    fn speed_divides_the_frames_before_rounding() {
        let logits = vec![20.0; 6];
        assert_eq!(
            decode_prosody_durations(&logits, 1, 6, 1.0).ok(),
            Some(vec![6])
        );
        assert_eq!(
            decode_prosody_durations(&logits, 1, 6, 2.0).ok(),
            Some(vec![3])
        );
        assert_eq!(
            decode_prosody_durations(&logits, 1, 6, 0.75).ok(),
            Some(vec![8])
        );
        // 5 / 2 = 2.5 rounds half to even: 2.
        let five = vec![20.0; 5];
        assert_eq!(
            decode_prosody_durations(&five, 1, 5, 2.0).ok(),
            Some(vec![2])
        );
    }

    #[test]
    fn a_bad_length_or_speed_is_refused() {
        assert!(decode_prosody_durations(&[0.0; 5], 2, 3, 1.0).is_err());
        assert!(decode_prosody_durations(&[0.0; 6], 2, 3, 0.0).is_err());
        assert!(decode_prosody_durations(&[0.0; 6], 2, 3, f32::NAN).is_err());
    }
}
