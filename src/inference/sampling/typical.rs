//! The typical-p filter over host logits.

/// Keep the most typical tokens up to `typical_p` of the probability mass.
///
/// A token's deviation is `|-ln(p) - H|`, with `H` the entropy of
/// `softmax(logits)`. Tokens join in ascending deviation, ties in id order.
/// The scan stops before a token once the kept mass reaches `typical_p`.
/// The first token always stays. Every other logit becomes `-inf`.
///
/// `typical_p >= 1` keeps every token, unless f32 rounding lifts the kept
/// mass to `typical_p` before the last one. A `typical_p` at or below 0
/// keeps one token.
pub fn apply_typical_filter(logits: &mut [f32], typical_p: f32) {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();

    let mut entropy = 0.0f32;
    for &p in &probs {
        let normalized = p / sum;
        if normalized > 0.0 {
            entropy -= normalized * normalized.ln();
        }
    }

    let mut deviations: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let normalized = p / sum;
            let info = if normalized > 0.0 {
                -normalized.ln()
            } else {
                f32::INFINITY
            };
            (i, (info - entropy).abs(), normalized)
        })
        .collect();
    deviations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut keep = vec![false; logits.len()];
    for &(idx, _, prob) in &deviations {
        if cumsum >= typical_p && cumsum > 0.0 {
            break;
        }
        keep[idx] = true;
        cumsum += prob;
    }
    for (logit, kept) in logits.iter_mut().zip(keep) {
        if !kept {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kept(logits: &[f32]) -> Vec<usize> {
        (0..logits.len())
            .filter(|&i| logits[i].is_finite())
            .collect()
    }

    #[test]
    fn p_one_keeps_every_token() {
        let original = [2.0f32, -1.0, 0.5, 0.5, 3.0];
        let mut logits = original;
        apply_typical_filter(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn a_uniform_row_keeps_every_token_at_p_one() {
        let mut logits = [0.25f32; 8];
        apply_typical_filter(&mut logits, 1.0);
        assert_eq!(kept(&logits), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_uniform_row_keeps_ties_in_id_order_up_to_p() {
        let mut logits = [0.0f32; 8];
        apply_typical_filter(&mut logits, 0.5);
        assert_eq!(kept(&logits), vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_small_p_keeps_the_most_typical_token() {
        // Three likely tokens and one rare one: the rare one is least typical.
        let mut logits = [1.0f32, 1.0, 1.0, -6.0];
        apply_typical_filter(&mut logits, 0.0);
        assert_eq!(kept(&logits).len(), 1);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }
}
