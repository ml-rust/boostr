//! Per-input-channel AWQ-style smoothing scale.
//!
//! For input channel `j`:
//!
//! ```text
//! s_j = a_j^alpha / w_j^(1 - alpha)
//! ```
//!
//! `a_j` is the RMS activation, `sqrt(mean_square_j)`, read from an
//! [`ImportanceEntry`](crate::quant::ImportanceEntry). `w_j` is the column's
//! weight magnitude summary: the max absolute value over column `j` of the
//! `[out_features, in_features]` weight. `alpha` trades the two off; `0`
//! leans entirely on the weight, `1` entirely on the activation.
//!
//! The raw `s_j` is then NORMALIZED so its geometric mean is 1, computed in
//! log space to avoid overflow on a long column. Normalizing keeps a whole
//! tensor's scale from drifting the block scales the quantizer picks
//! systematically up or down; only the RELATIVE weight between channels is
//! the point of this transform.
//!
//! # `alpha == 0.0` is special-cased
//!
//! The general formula does not reduce to `s_j = 1` at `alpha = 0`: it
//! reduces to `s_j = 1 / w_j`, and after normalization that is
//! `s_j = geometric_mean(w) / w_j` — one for every column only if every
//! column's weight magnitude is identical. So `alpha == 0.0` is special-cased
//! here to return an all-ones vector directly, which is what makes the
//! probe's control path (`alpha = 0`) bit-identical to no smoothing at all.
//!
//! # Absent is not zero
//!
//! A channel with NO importance entry is a different situation than what
//! this function handles: that is the whole TENSOR being skipped by the
//! caller before this function is ever called (see
//! [`ImportanceMatrix::get`](crate::quant::ImportanceMatrix::get)). What this
//! function handles is a channel that DOES have an entry, whose value for
//! that one channel is zero, non-finite, or paired with a zero/non-finite
//! weight column — see "Degenerate channels" below.
//!
//! # Degenerate channels
//!
//! A channel whose activation importance is zero or non-finite, or whose
//! weight column's magnitude is zero or non-finite, gets `s_j = 1.0` exactly
//! — never `0`, which the divide in the caller's reconstruction step cannot
//! survive, and never infinity. Such a channel is excluded from the
//! normalization's geometric mean, so it never skews the scale the other
//! columns receive, and it is force-set to `1.0` AFTER normalization, so it
//! is never perturbed by it.
//!
//! Every returned `s_j` is finite and strictly positive, unconditionally.
//!
//! The normalization and the degenerate-channel rule are shared with
//! [`super::weight_only::weight_only_smoothing_scale`] via
//! `super::normalize` — one copy of that logic, not two.

use super::normalize::{column_max_abs, normalize_to_unit_geometric_mean};

/// Per-input-channel smoothing scale for a `[out_features, in_features]`
/// weight.
///
/// - `activation_mean_square`: `mean(x_j^2)` per input channel, length
///   `in_features` — [`ImportanceEntry::mean_square`](crate::quant::ImportanceEntry::mean_square)'s
///   output for the tensor being smoothed.
/// - `weight`: the weight's own values, row-major, length
///   `out_features * in_features`.
/// - `in_features`: the row width both slices are keyed by.
/// - `alpha`: in `[0, 1]`. `0` returns all ones (see the module docs); `1`
///   ignores the weight column's magnitude entirely.
///
/// A malformed input — `activation_mean_square.len() != in_features`, or a
/// `weight` whose length is not a multiple of `in_features` — returns an
/// all-ones vector rather than panicking: the safe, no-op scale for a
/// pairing this function cannot make sense of.
///
/// Returns a `Vec<f32>` of length `in_features`, every entry finite and
/// strictly positive.
pub fn smoothing_scale(
    activation_mean_square: &[f32],
    weight: &[f32],
    in_features: usize,
    alpha: f32,
) -> Vec<f32> {
    if in_features == 0 {
        return Vec::new();
    }
    if activation_mean_square.len() != in_features || !weight.len().is_multiple_of(in_features) {
        return vec![1.0; activation_mean_square.len()];
    }
    if alpha == 0.0 {
        return vec![1.0; in_features];
    }

    let weight_max_abs = column_max_abs(weight, in_features);

    // `None` marks a degenerate channel: excluded from the geometric mean,
    // forced to exactly 1.0 in the output.
    let raw: Vec<Option<f64>> = activation_mean_square
        .iter()
        .zip(weight_max_abs.iter())
        .map(|(&mean_square, &w_max)| {
            if !(mean_square.is_finite() && mean_square > 0.0 && w_max.is_finite() && w_max > 0.0) {
                return None;
            }
            let a_rms = f64::from(mean_square).sqrt();
            let alpha = f64::from(alpha);
            let value = a_rms.powf(alpha) / f64::from(w_max).powf(1.0 - alpha);
            if value.is_finite() && value > 0.0 {
                Some(value)
            } else {
                None
            }
        })
        .collect();

    normalize_to_unit_geometric_mean(raw)
}

#[cfg(test)]
mod tests {
    use super::smoothing_scale;

    /// Geometric mean of a slice, for asserting the normalization held.
    fn geometric_mean(values: &[f32]) -> f64 {
        let log_sum: f64 = values.iter().map(|&v| f64::from(v).ln()).sum();
        (log_sum / values.len() as f64).exp()
    }

    #[test]
    fn alpha_zero_is_all_ones() {
        let activation = [0.1f32, 4.0, 100.0, 0.0];
        let weight = [1.0f32, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
        let s = smoothing_scale(&activation, &weight, 4, 0.0);
        assert_eq!(s, vec![1.0f32; 4]);
    }

    #[test]
    fn geometric_mean_of_result_is_one() {
        // No degenerate channel here, so the normalization's own invariant
        // is exact up to float error.
        let activation = [0.5f32, 2.0, 8.0, 0.25];
        let weight = [1.0f32, 2.0, 0.5, 4.0, 3.0, 1.0, 2.0, 0.5];
        let s = smoothing_scale(&activation, &weight, 4, 0.5);
        let gm = geometric_mean(&s);
        assert!((gm - 1.0).abs() < 1e-5, "geometric mean was {gm}");
    }

    #[test]
    fn zero_importance_channel_is_exactly_one() {
        let activation = [0.0f32, 4.0, 9.0];
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = smoothing_scale(&activation, &weight, 3, 0.5);
        assert_eq!(s[0], 1.0);
    }

    #[test]
    fn zero_weight_column_is_exactly_one() {
        let activation = [1.0f32, 4.0, 9.0];
        // Column 1 is all zero.
        let weight = [1.0f32, 0.0, 3.0, 4.0, 0.0, 6.0];
        let s = smoothing_scale(&activation, &weight, 3, 0.5);
        assert_eq!(s[1], 1.0);
    }

    #[test]
    fn extreme_values_stay_finite_and_positive() {
        let activation = [f32::MIN_POSITIVE, 1e30, f32::NAN, f32::INFINITY, 1.0];
        let weight = [
            1e-30f32,
            1e30,
            1.0,
            1.0,
            1.0,
            -1e20,
            f32::NAN,
            2.0,
            f32::INFINITY,
            3.0,
        ];
        let s = smoothing_scale(&activation, &weight, 5, 0.5);
        assert_eq!(s.len(), 5);
        for value in s {
            assert!(value.is_finite(), "{value} is not finite");
            assert!(value > 0.0, "{value} is not strictly positive");
        }
    }

    #[test]
    fn larger_activation_gets_larger_scale_at_half_alpha() {
        let activation = [1.0f32, 100.0];
        let weight = [1.0f32, 1.0, 1.0, 1.0];
        let s = smoothing_scale(&activation, &weight, 2, 0.5);
        assert!(s[1] > s[0], "s = {s:?}");
    }
}
