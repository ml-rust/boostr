//! Shared temperature/top-k/top-p/min-p/multinomial math for `sample_token`
//! and `logits_to_token`'s stochastic branch.

/// Scales `logits` by `1 / temperature` in place, then runs softmax →
/// top-k → top-p → min-p filtering.
///
/// Returns the surviving `(index, probability)` pairs sorted by
/// probability, descending.
pub(super) fn filter_distribution(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
) -> Vec<(usize, f32)> {
    if temperature != 1.0 {
        let inv_temp = 1.0 / temperature;
        for l in logits.iter_mut() {
            *l *= inv_temp;
        }
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    if top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum > top_p {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    if min_p > 0.0 && !indexed.is_empty() {
        let max_prob = indexed[0].1;
        let threshold = min_p * max_prob;
        indexed.retain(|(_, p)| *p >= threshold);
    }

    indexed
}

/// Renormalizes `indexed` and draws the token whose cumulative probability
/// first exceeds `random_val`.
///
/// Falls back to the last (lowest-probability) survivor if none does —
/// guards float rounding at the tail of the cumulative sum.
pub(super) fn sample_from_distribution(indexed: &[(usize, f32)], random_val: f32) -> usize {
    let total: f32 = indexed.iter().map(|(_, p)| p).sum();
    let mut cumsum = 0.0f32;
    for (i, p) in indexed {
        cumsum += p / total;
        if cumsum > random_val {
            return *i;
        }
    }
    indexed.last().map(|(i, _)| *i).unwrap_or(0)
}
