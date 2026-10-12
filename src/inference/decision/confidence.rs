//! Confidence scalars of Choice and Score answers.
//!
//! Inputs are distributions that the caller has already checked.

/// The Score confidence rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ScoreConfidence {
    /// `max(0, 1 - E|i - mode| / D)`, mode = first most likely level, and
    /// `D = mean_i |i - (L - 1) / 2|`: the mean distance of the levels from the
    /// middle of the scale. Level distances use the level index, not the value.
    #[default]
    Distance,
    /// The Choice rule `(p_max - 1/L) / (1 - 1/L)` applied to the levels.
    ChanceCorrected,
}

/// The index of the first largest entry. `probs` must not be empty.
pub fn first_argmax(probs: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in probs.iter().enumerate() {
        if v > probs[best] {
            best = i;
        }
    }
    best
}

/// `(p_max - 1/K) / (1 - 1/K)` clipped to `[0, 1]`; 1 when K = 1.
pub fn choice_confidence(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k <= 1 {
        return 1.0;
    }
    let p_max = probs[first_argmax(probs)];
    let chance = 1.0 / k as f64;
    ((p_max - chance) / (1.0 - chance)).clamp(0.0, 1.0)
}

/// The Score confidence of a level distribution under `rule`; 1 when L = 1.
pub fn score_confidence(probs: &[f64], rule: ScoreConfidence) -> f64 {
    let l = probs.len();
    if l <= 1 {
        return 1.0;
    }
    match rule {
        ScoreConfidence::Distance => {
            let mode = first_argmax(probs) as f64;
            let mid = (l as f64 - 1.0) / 2.0;
            let d = (0..l).map(|i| (i as f64 - mid).abs()).sum::<f64>() / l as f64;
            let spread: f64 = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| p * (i as f64 - mode).abs())
                .sum();
            (1.0 - spread / d).clamp(0.0, 1.0)
        }
        ScoreConfidence::ChanceCorrected => choice_confidence(probs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_argmax_takes_the_first_tie() {
        assert_eq!(first_argmax(&[0.4, 0.4, 0.2]), 0);
        assert_eq!(first_argmax(&[0.2, 0.4, 0.4]), 1);
    }

    #[test]
    fn choice_confidence_ends() {
        assert_eq!(choice_confidence(&[0.25; 4]), 0.0);
        assert_eq!(choice_confidence(&[0.0, 1.0, 0.0]), 1.0);
        assert_eq!(choice_confidence(&[1.0]), 1.0);
        // K = 4, p_max = 0.5: (0.5 - 0.25) / 0.75 = 1/3.
        assert!((choice_confidence(&[0.5, 0.2, 0.2, 0.1]) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn distance_confidence_hand_values() {
        let d = ScoreConfidence::Distance;
        // L = 3: D = (1 + 0 + 1) / 3 = 2/3. Uniform: mode 0, spread = 1/3 + 2/3 = 1 -> max(0, 1 - 1.5) = 0.
        assert_eq!(score_confidence(&[1.0 / 3.0; 3], d), 0.0);
        assert_eq!(score_confidence(&[0.0, 1.0, 0.0], d), 1.0);
        // L = 3, p = [0.1, 0.8, 0.1]: mode 1, spread 0.2 -> 1 - 0.2 / (2/3) = 0.7.
        assert!((score_confidence(&[0.1, 0.8, 0.1], d) - 0.7).abs() < 1e-12);
        // L = 5: D = (2 + 1 + 0 + 1 + 2) / 5 = 1.2. p = [0, 0.5, 0.5, 0, 0]: mode 1, spread 0.5.
        assert!(
            (score_confidence(&[0.0, 0.5, 0.5, 0.0, 0.0], d) - (1.0 - 0.5 / 1.2)).abs() < 1e-12
        );
        assert_eq!(score_confidence(&[1.0], d), 1.0);
    }

    #[test]
    fn two_levels_agree_across_rules() {
        let p = [0.3, 0.7];
        let a = score_confidence(&p, ScoreConfidence::Distance);
        let b = score_confidence(&p, ScoreConfidence::ChanceCorrected);
        assert!((a - 0.4).abs() < 1e-12);
        assert!((b - 0.4).abs() < 1e-12);
    }

    #[test]
    fn chance_corrected_score() {
        let p = [0.1, 0.8, 0.1];
        let c = score_confidence(&p, ScoreConfidence::ChanceCorrected);
        assert!((c - (0.8 - 1.0 / 3.0) / (2.0 / 3.0)).abs() < 1e-12);
    }
}
