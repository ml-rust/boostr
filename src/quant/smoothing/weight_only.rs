//! Calibration-free per-input-channel weight smoothing: a scale derived
//! ENTIRELY from the weight's own column magnitudes, needing no activation
//! statistic and no importance matrix.
//!
//! For input channel `j`:
//!
//! ```text
//! s_j = (1 / w_j) ^ alpha,  then normalized to geometric mean 1
//! ```
//!
//! `w_j` is the same per-column max-absolute-value summary
//! [`super::scale::smoothing_scale`] uses. `alpha = 0` returns all ones — the
//! same control the activation-derived source returns at `alpha = 0`,
//! bit-exact. `alpha = 1` gives full column equalization: after
//! normalization, every non-degenerate column of `W * s` has the same
//! magnitude.
//!
//! This is what the general activation formula collapses to when the
//! activation term drops out entirely — expressed directly here, as its own
//! closed form, rather than by feeding a dummy activation into
//! [`super::scale::smoothing_scale`]. `1 - alpha` reaching `0` there would
//! need a `pow` of the weight term to the power `0` right where `alpha = 1`
//! is supposed to matter most; this form has no such vanishing exponent.
//!
//! # MEASURED: this source HURTS. Do not reach for it.
//!
//! Every non-zero `alpha` scored worse than the `alpha = 0` control, on both
//! evaluation splits, monotonically in `alpha`. `alpha = 1` was catastrophic.
//!
//! Mechanism: `1 / w_j` is unbounded. A near-zero column yields a huge scale.
//! A block scale takes the largest magnitude in its block, so one inflated
//! column destroys every weight sharing that block.
//! [`super::scale::smoothing_scale`] escapes this because an activation RMS is
//! bounded.
//!
//! Kept as the recorded answer to a question worth asking once.
//!
//! # An importance matrix still selects the tensor SET, never the scale
//!
//! A caller pairing this source with `--smooth-imatrix` still transforms
//! only tensors that have an importance entry — that is what keeps the
//! transformed tensor SET identical between an activation-derived run and a
//! weight-only run of the same command line, so the two are directly
//! comparable. The matrix contributes NOTHING to the scale computed here:
//! this function never reads it and never reads any activation statistic.
//!
//! # Same invariants as the activation-derived source
//!
//! Every returned value is finite and strictly positive, unconditionally. A
//! channel whose weight column magnitude is zero or non-finite gets `s_j =
//! 1.0` exactly, is excluded from the geometric mean, and is set AFTER
//! normalization so it is never perturbed by it — see
//! [`super::normalize::normalize_to_unit_geometric_mean`], which this
//! function shares with [`super::scale::smoothing_scale`] rather than
//! reimplementing.

use super::normalize::{column_max_abs, normalize_to_unit_geometric_mean};

/// Weight-only per-input-channel smoothing scale for a
/// `[out_features, in_features]` weight. Needs no activation data.
///
/// - `weight`: the weight's own values, row-major, length
///   `out_features * in_features`.
/// - `in_features`: the row width `weight` is keyed by.
/// - `alpha`: in `[0, 1]`. `0` returns all ones (see the module docs); `1`
///   equalizes every non-degenerate column's magnitude.
///
/// A `weight` whose length is not a multiple of `in_features` returns an
/// all-ones vector of length `in_features` rather than panicking: the safe,
/// no-op scale for an input this function cannot make sense of.
///
/// Returns a `Vec<f32>` of length `in_features`, every entry finite and
/// strictly positive.
pub fn weight_only_smoothing_scale(weight: &[f32], in_features: usize, alpha: f32) -> Vec<f32> {
    if in_features == 0 {
        return Vec::new();
    }
    if !weight.len().is_multiple_of(in_features) {
        return vec![1.0; in_features];
    }
    if alpha == 0.0 {
        return vec![1.0; in_features];
    }

    let weight_max_abs = column_max_abs(weight, in_features);
    let alpha = f64::from(alpha);

    // `None` marks a degenerate channel: excluded from the geometric mean,
    // forced to exactly 1.0 in the output.
    let raw: Vec<Option<f64>> = weight_max_abs
        .iter()
        .map(|&w_max| {
            if !(w_max.is_finite() && w_max > 0.0) {
                return None;
            }
            let value = (1.0 / f64::from(w_max)).powf(alpha);
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
    use super::weight_only_smoothing_scale;

    /// Geometric mean of a slice, for asserting the normalization held.
    fn geometric_mean(values: &[f32]) -> f64 {
        let log_sum: f64 = values.iter().map(|&v| f64::from(v).ln()).sum();
        (log_sum / values.len() as f64).exp()
    }

    #[test]
    fn alpha_zero_is_all_ones() {
        let weight = [1.0f32, 2.0, 3.0, 4.0, -5.0, 6.0];
        let s = weight_only_smoothing_scale(&weight, 3, 0.0);
        assert_eq!(s, vec![1.0f32; 3]);
    }

    #[test]
    fn alpha_one_equalizes_every_column() {
        // Shape [2, 3]: row0 = [1, 2, 4], row1 = [3, 6, 8].
        // Column max-abs: [3, 6, 8] — three distinct magnitudes.
        let weight = [1.0f32, 2.0, 4.0, 3.0, 6.0, 8.0];
        let column_max_abs = [3.0f64, 6.0, 8.0];
        let s = weight_only_smoothing_scale(&weight, 3, 1.0);

        let products: Vec<f64> = column_max_abs
            .iter()
            .zip(s.iter())
            .map(|(&w, &si)| w * f64::from(si))
            .collect();
        for product in &products[1..] {
            assert!(
                (product - products[0]).abs() < 1e-4,
                "columns did not equalize: {products:?}"
            );
        }
    }

    #[test]
    fn geometric_mean_of_result_is_one() {
        let weight = [1.0f32, 2.0, 0.5, 4.0, 3.0, 1.0, 2.0, 0.5];
        let s = weight_only_smoothing_scale(&weight, 4, 0.5);
        let gm = geometric_mean(&s);
        assert!((gm - 1.0).abs() < 1e-5, "geometric mean was {gm}");
    }

    #[test]
    fn zero_magnitude_column_is_exactly_one() {
        // Column 1 is all zero.
        let weight = [1.0f32, 0.0, 3.0, 4.0, 0.0, 6.0];
        let s = weight_only_smoothing_scale(&weight, 3, 0.5);
        assert_eq!(s[1], 1.0);
    }

    #[test]
    fn extreme_values_stay_finite_and_positive() {
        let weight = [
            1e-30f32,
            1e30,
            1.0,
            1.0,
            -1e20,
            f32::NAN,
            2.0,
            f32::INFINITY,
            3.0,
        ];
        let s = weight_only_smoothing_scale(&weight, 3, 0.5);
        assert_eq!(s.len(), 3);
        for value in s {
            assert!(value.is_finite(), "{value} is not finite");
            assert!(value > 0.0, "{value} is not strictly positive");
        }
    }
}
