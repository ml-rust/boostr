//! One group's scale search and codebook assignment. Same candidate-scale
//! SHAPE `tcf-core`'s symmetric search uses
//! (`hats/tcf/tcf-core/src/encoding/quantize.rs::fit_symmetric_scale`), with
//! no least-squares refit step — the levels here are not equally spaced, so
//! the refit's linear normal equation does not apply. Every candidate is
//! scored by direct weighted squared error instead.

use super::levels::Codebook;

/// Both codebooks are normalized so the largest-magnitude level is exactly
/// `1.0` (see `levels.rs`) — this is the `qmax` the candidate-scale search
/// sweeps around, independent of which codebook is active.
const QMAX: f32 = 1.0;

/// The candidate multiplier sweep: `-9..=9` in steps of `0.1`, the same
/// shape `tcf-core`'s `SearchEffort::Standard` uses. Each yields a candidate
/// scale `d = max_abs / (QMAX + multiplier)`.
pub(super) fn candidate_multipliers() -> impl Iterator<Item = f32> {
    (-9..=9).map(|is| 0.1 * is as f32)
}

/// The codebook level closest to `v`, by absolute difference. Ties (equal
/// distance to two levels) resolve to the level appearing FIRST in the
/// ascending array, so the result never depends on float rounding order.
pub(super) fn nearest_level(v: f32, levels: &[f32; 16]) -> f32 {
    let mut best_index = 0usize;
    let mut best_diff = f32::INFINITY;
    for (index, &level) in levels.iter().enumerate() {
        let diff = (v - level).abs();
        if diff < best_diff {
            best_diff = diff;
            best_index = index;
        }
    }
    levels[best_index]
}

/// Weighted squared reconstruction error `sum(w_i * (x_i - d * l_i)^2)` for
/// `values` against candidate scale `d`, where each `l_i` is the nearest
/// codebook level to `x_i / d`. `d` MUST be finite and strictly positive.
fn weighted_squared_error(values: &[f32], weights: &[f32], d: f32, levels: &[f32; 16]) -> f64 {
    let mut err = 0.0f64;
    for (index, &x) in values.iter().enumerate() {
        let level = nearest_level(x / d, levels);
        let diff = f64::from(x) - f64::from(d) * f64::from(level);
        let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
        err += w * diff * diff;
    }
    err
}

/// Quantizes then dequantizes one group of up to 32 weights against
/// `codebook`, returning the reconstructed values (same length as `values`).
///
/// - `weights[i]` scores element `i`'s contribution; a short `weights` slice
///   treats a missing entry as `1.0`, never panics.
/// - An all-zero group returns exact zeros: `max_abs == 0.0` has no live
///   scale, so it is handled directly rather than searched.
/// - Otherwise: try every candidate `d = max_abs / (QMAX + multiplier)` for
///   `multiplier` in [`candidate_multipliers`], skipping one that is
///   non-finite or non-positive; score by [`weighted_squared_error`]; keep
///   the strictly-lower-error candidate, so an exact tie keeps the FIRST
///   (iteration order, ascending multiplier).
/// - If every candidate's `d` is skipped (degenerate `max_abs`), returns
///   exact zeros — the same safe fallback as the all-zero case.
pub(super) fn quantize_group(values: &[f32], weights: &[f32], codebook: Codebook) -> Vec<f32> {
    let levels = codebook.levels();
    let max_abs = values.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return vec![0.0f32; values.len()];
    }

    let mut best: Option<(f64, f32)> = None;
    for multiplier in candidate_multipliers() {
        let d = max_abs / (QMAX + multiplier);
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        let err = weighted_squared_error(values, weights, d, levels);
        let improves = match best {
            None => true,
            Some((best_err, _)) => err < best_err,
        };
        if improves {
            best = Some((err, d));
        }
    }

    let Some((_, d)) = best else {
        return vec![0.0f32; values.len()];
    };
    values
        .iter()
        .map(|&x| d * nearest_level(x / d, levels))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_group_round_trips_to_exact_zeros() {
        let values = [0.0f32; 5];
        let weights = [1.0f32; 5];
        let out = quantize_group(&values, &weights, Codebook::Uniform);
        assert_eq!(out, vec![0.0f32; 5]);
    }

    #[test]
    fn values_already_on_codebook_levels_round_trip_near_exact() {
        for codebook in [Codebook::Uniform, Codebook::Nf4] {
            let levels = codebook.levels();
            // d0 = 2.0; values sit exactly on d0 * level for several levels,
            // including the max-magnitude one, so max_abs == d0 exactly and
            // the zero-multiplier candidate (is = 0) reproduces d0 exactly.
            let d0 = 2.0f32;
            let picks = [15usize, 0, 8, 3, 12];
            let values: Vec<f32> = picks.iter().map(|&i| d0 * levels[i]).collect();
            let weights = vec![1.0f32; values.len()];
            let out = quantize_group(&values, &weights, codebook);
            for (got, want) in out.iter().zip(values.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "{codebook:?}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn short_weights_slice_treats_missing_entries_as_one() {
        let values = [0.6f32, -0.2, 0.05];
        let out_short = quantize_group(&values, &[2.0f32], Codebook::Uniform);
        let out_full = quantize_group(&values, &[2.0f32, 1.0, 1.0], Codebook::Uniform);
        assert_eq!(out_short, out_full);
    }

    /// A mathematical property of weighted-sum minimization over a FIXED
    /// discrete candidate set, not an empirical coincidence: if `d*`
    /// minimizes at weight `w` and `d**` minimizes at a larger weight `w'`,
    /// optimality of each at its own weight forces
    /// `(w - w') * (e0(d*) - e0(d**)) <= 0`, and `w < w'` makes that
    /// `e0(d*) >= e0(d**)` — element 0's error at the CHOSEN scale can never
    /// rise as its own weight rises, regardless of the other elements.
    #[test]
    fn raising_one_elements_weight_never_increases_its_own_error() {
        let values = [0.61f32, -0.23, 0.05, 0.9, -0.75, 0.12, 0.33, -0.44];
        let mut prev_err = f32::INFINITY;
        for &w0 in &[1.0f32, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0] {
            let mut weights = vec![1.0f32; values.len()];
            weights[0] = w0;
            let out = quantize_group(&values, &weights, Codebook::Nf4);
            let err = (values[0] - out[0]).abs();
            assert!(
                err <= prev_err + 1e-6,
                "error rose at w0={w0}: prev={prev_err}, now={err}"
            );
            prev_err = err;
        }
    }
}
