//! Shared normalization for every smoothing scale source in this module:
//! normalize a per-channel raw scale to geometric mean 1 in log space, and
//! force a degenerate channel to exactly `1.0`.
//!
//! One copy of this rule so every source — activation-derived, weight-only,
//! and any added later — treats a degenerate channel identically and
//! normalizes the same way. Neither helper reads an importance matrix or an
//! activation statistic; both work on whatever raw values their caller
//! already derived.

/// The largest absolute value in column `j`, for every `j`, of a row-major
/// `[out_features, in_features]` weight.
///
/// `weight.len()` MUST be a multiple of `in_features`; every caller checks
/// this before calling.
pub(super) fn column_max_abs(weight: &[f32], in_features: usize) -> Vec<f32> {
    let out_features = weight.len() / in_features;
    let mut max_abs = vec![0f32; in_features];
    for row in 0..out_features {
        let base = row * in_features;
        for (j, slot) in max_abs.iter_mut().enumerate() {
            let v = weight[base + j].abs();
            if v > *slot {
                *slot = v;
            }
        }
    }
    max_abs
}

/// Normalize `raw` — one entry per channel, `None` marking a channel its
/// caller has already flagged degenerate — so the geometric mean of the
/// VALID entries is 1, computed in log space to avoid overflow on a long
/// column.
///
/// Every degenerate channel is then forced to exactly `1.0`, unconditionally:
/// it is excluded from the geometric mean so it never skews the scale other
/// channels receive, and it is set AFTER normalization so it is never itself
/// perturbed by it.
///
/// Returns a vector the same length as `raw`, every entry finite and
/// strictly positive.
pub(super) fn normalize_to_unit_geometric_mean(raw: Vec<Option<f64>>) -> Vec<f32> {
    let log_sum: f64 = raw.iter().filter_map(|v| v.map(f64::ln)).sum();
    let valid_count = raw.iter().filter(|v| v.is_some()).count();
    if valid_count == 0 {
        return vec![1.0; raw.len()];
    }
    let log_gm = log_sum / valid_count as f64;
    raw.into_iter()
        .map(|v| match v {
            Some(value) => (value.ln() - log_gm).exp() as f32,
            None => 1.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::normalize_to_unit_geometric_mean;

    #[test]
    fn all_degenerate_channels_are_all_ones() {
        let raw: Vec<Option<f64>> = vec![None, None, None];
        assert_eq!(normalize_to_unit_geometric_mean(raw), vec![1.0f32; 3]);
    }

    #[test]
    fn a_mix_of_degenerate_and_valid_stays_finite_and_positive() {
        let raw: Vec<Option<f64>> = vec![None, Some(2.0), Some(0.5), None];
        for value in normalize_to_unit_geometric_mean(raw) {
            assert!(value.is_finite());
            assert!(value > 0.0);
        }
    }
}
