//! Affine variant of the codebook probe: reconstruction is `m + d *
//! levels()[code]`, a per-group MINIMUM plus scale, instead of the
//! symmetric arm's `d * level`. Fills the AFFINE + NON-UNIFORM cell no GGUF
//! format occupies today — K-quants are affine with uniform levels, IQ
//! formats are non-uniform but symmetric.
//!
//! Reuses the symmetric arm's candidate-scale sweep
//! ([`candidate_multipliers`]) unchanged, and applies the SAME first-on-tie
//! nearest-level rule via [`nearest_code`] (index instead of value, so the
//! refit step can look codes back up), so the two arms differ ONLY in level
//! placement and in fitting a minimum — never in search effort.

use super::levels::NF4_LEVELS;
use super::quantize::candidate_multipliers;
use super::roundtrip::GROUP_SIZE;

/// Which 16-level AFFINE codebook a group quantizes against. Levels span
/// `[0, 1]`; a group reconstructs to `m + d * levels()[code]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AffineCodebook {
    /// Evenly spaced affine 4-bit grid — the CONTROL, the same shape Q4_1
    /// and Q4_K use.
    Uniform,
    /// The symmetric [`NF4_LEVELS`] shifted onto `[0, 1]`. After an affine
    /// shift, a group's mean sits mid-span rather than at an edge (as a
    /// zero-centered symmetric fit forces), so density concentrates near
    /// the middle of `[0, 1]` — exactly where NF4's quantile spacing already
    /// concentrates its levels. Shifting NF4 targets that density directly,
    /// instead of building a fresh affine quantile table from scratch.
    Nf4Shifted,
}

/// `i / 15` for `i` in `0..16`: 16 evenly spaced levels on `[0, 1]`.
const fn uniform_affine_levels() -> [f32; 16] {
    let mut out = [0.0f32; 16];
    let mut i = 0usize;
    while i < 16 {
        out[i] = i as f32 / 15.0;
        i += 1;
    }
    out
}

/// [`NF4_LEVELS`] mapped from `[-1, 1]` onto `[0, 1]` by `(v + 1) / 2`.
const fn nf4_shifted_levels() -> [f32; 16] {
    let mut out = [0.0f32; 16];
    let mut i = 0usize;
    while i < 16 {
        out[i] = (NF4_LEVELS[i] + 1.0) / 2.0;
        i += 1;
    }
    out
}

/// 16 evenly spaced levels on `[0, 1]` — the affine CONTROL grid.
pub const AFFINE_UNIFORM_LEVELS: [f32; 16] = uniform_affine_levels();
/// [`NF4_LEVELS`] shifted onto `[0, 1]` — derived, never a second copy of
/// the quantile numbers.
pub const NF4_SHIFTED_LEVELS: [f32; 16] = nf4_shifted_levels();

impl AffineCodebook {
    /// The 16 reconstruction levels for this codebook, ascending, spanning
    /// `[0, 1]` — the value a group reconstructs to is `m + d * level`.
    pub fn levels(self) -> &'static [f32; 16] {
        match self {
            AffineCodebook::Uniform => &AFFINE_UNIFORM_LEVELS,
            AffineCodebook::Nf4Shifted => &NF4_SHIFTED_LEVELS,
        }
    }
}

/// Below this span, a group is treated as constant: no live scale to search
/// over, and dividing by a near-zero span would blow up the fit.
const SPAN_EPS: f32 = 1e-6;

/// The affine scale sweep at unit `qmax = 1.0`, the SAME shape the symmetric
/// arm uses (see [`candidate_multipliers`]): `d = span / (1.0 + multiplier)`.
fn candidate_scales(span: f32) -> impl Iterator<Item = f32> {
    candidate_multipliers().map(move |multiplier| span / (1.0 + multiplier))
}

/// Nearest-level CODE (its index) for `u` under `levels`, first-on-tie —
/// same tie rule as the symmetric arm's `nearest_level`, returning the
/// index instead of the value so the caller can refit `(d, m)` against the
/// chosen codes.
fn nearest_code(u: f32, levels: &[f32; 16]) -> usize {
    let mut best_index = 0usize;
    let mut best_diff = f32::INFINITY;
    for (index, &level) in levels.iter().enumerate() {
        let diff = (u - level).abs();
        if diff < best_diff {
            best_diff = diff;
            best_index = index;
        }
    }
    best_index
}

/// Weighted squared error of reconstructing `values` as `m + d * level`,
/// where `level` is the nearest codebook entry to `(x - m) / d`.
fn weighted_squared_error_affine(
    values: &[f32],
    weights: &[f32],
    d: f32,
    m: f32,
    levels: &[f32; 16],
) -> f64 {
    let mut err = 0.0f64;
    for (index, &x) in values.iter().enumerate() {
        let code = nearest_code((x - m) / d, levels);
        let diff = f64::from(x) - f64::from(m) - f64::from(d) * f64::from(levels[code]);
        let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
        err += w * diff * diff;
    }
    err
}

/// One weighted least-squares refit of `(d, m)` against `codes` (already
/// fixed by a prior nearest-level search): solves the 2x2 normal equations
/// for `y = m + d*u` over `(u_i, y_i, w_i)`. Runs ONCE — it does not iterate
/// to convergence, since it refits against codes that are already chosen.
///
/// Returns `None` when the system is not solvable (every code identical, or
/// the weights are degenerate), telling the caller to keep the candidate
/// `(d, m)` it already has.
fn refit_affine(
    values: &[f32],
    weights: &[f32],
    codes: &[usize],
    levels: &[f32; 16],
) -> Option<(f32, f32)> {
    let mut sw = 0.0f64;
    let mut su = 0.0f64;
    let mut suu = 0.0f64;
    let mut sy = 0.0f64;
    let mut suy = 0.0f64;
    for (index, &x) in values.iter().enumerate() {
        let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
        let u = f64::from(levels[codes[index]]);
        let y = f64::from(x);
        sw += w;
        su += w * u;
        suu += w * u * u;
        sy += w * y;
        suy += w * u * y;
    }
    let denom = sw * suu - su * su;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return None;
    }
    let d = (sw * suy - su * sy) / denom;
    let m = (sy - d * su) / sw;
    if !d.is_finite() || !m.is_finite() {
        return None;
    }
    Some((d as f32, m as f32))
}

/// Quantizes then dequantizes one group of up to [`GROUP_SIZE`] weights
/// against `codebook`'s affine reconstruction `m + d * levels()[code]`.
///
/// - `weights[i]` scores element `i`; a short `weights` slice treats a
///   missing entry as `1.0`, same rule as the symmetric arm.
/// - A degenerate group (`span < SPAN_EPS`, or a non-finite `min`/`span`)
///   stores `d = 0`, `m = min`, every code `0`, and reconstructs every
///   element to exactly `m` — the affine fit's exact answer for a constant
///   group, which a symmetric zero-centered fit cannot represent without an
///   offset.
/// - Otherwise: sweep [`candidate_scales`] around `span`, skipping a
///   non-finite or non-positive `d`; for each, assign every value its
///   nearest level under `m + d*u` (`m` fixed at `min`) and score by
///   weighted squared error; keep the strictly-lower-error candidate
///   (first-on-tie, matching the symmetric arm).
/// - After the sweep: [`refit_affine`] once against the winning candidate's
///   codes, then re-assigns codes against the refit `(d, m)` before
///   reconstructing. A refit that is not solvable, or lands on a
///   non-positive `d`, is discarded and the candidate `(d, m)` is kept.
fn quantize_group_affine(values: &[f32], weights: &[f32], codebook: AffineCodebook) -> Vec<f32> {
    let levels = codebook.levels();
    let min = values.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
    let max = values.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let span = max - min;

    if !min.is_finite() || !span.is_finite() || span < SPAN_EPS {
        return vec![min; values.len()];
    }

    let mut best: Option<(f64, f32)> = None;
    for d in candidate_scales(span) {
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        let err = weighted_squared_error_affine(values, weights, d, min, levels);
        let improves = match best {
            None => true,
            Some((best_err, _)) => err < best_err,
        };
        if improves {
            best = Some((err, d));
        }
    }

    let Some((_, mut d)) = best else {
        return vec![min; values.len()];
    };
    let mut m = min;
    let mut codes: Vec<usize> = values
        .iter()
        .map(|&x| nearest_code((x - m) / d, levels))
        .collect();

    if let Some((refit_d, refit_m)) = refit_affine(values, weights, &codes, levels)
        && refit_d.is_finite()
        && refit_d > 0.0
    {
        d = refit_d;
        m = refit_m;
        codes = values
            .iter()
            .map(|&x| nearest_code((x - m) / d, levels))
            .collect();
    }

    codes.iter().map(|&code| m + d * levels[code]).collect()
}

/// Quantizes then dequantizes every value in `values` against `codebook`,
/// grouping [`GROUP_SIZE`] consecutive elements per row exactly like
/// [`super::codebook_round_trip`] — same tail handling, same
/// never-crosses-a-row-boundary rule, same `in_features == 0` no-op. See
/// that function's docs for the full contract, which this mirrors unchanged.
pub fn affine_codebook_round_trip(
    values: &[f32],
    in_features: usize,
    codebook: AffineCodebook,
    weights: &[f32],
) -> Vec<f32> {
    if in_features == 0 {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(values.len());
    let value_rows = values.chunks(in_features);
    let weight_rows = weights.chunks(in_features);
    for (row, weight_row) in value_rows.zip(weight_rows) {
        for (group, weight_group) in row.chunks(GROUP_SIZE).zip(weight_row.chunks(GROUP_SIZE)) {
            out.extend(quantize_group_affine(group, weight_group, codebook));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::levels::Codebook;
    use super::super::quantize::quantize_group;
    use super::*;

    fn assert_16_from_zero_to_one(levels: &[f32; 16]) {
        assert_eq!(levels.len(), 16);
        assert_eq!(
            levels[0], 0.0,
            "first level was {}, expected 0.0",
            levels[0]
        );
        assert_eq!(
            levels[15], 1.0,
            "last level was {}, expected 1.0",
            levels[15]
        );
    }

    #[test]
    fn affine_uniform_has_16_entries_from_zero_to_one() {
        assert_16_from_zero_to_one(&AFFINE_UNIFORM_LEVELS);
    }

    #[test]
    fn nf4_shifted_has_16_entries_from_zero_to_one() {
        assert_16_from_zero_to_one(&NF4_SHIFTED_LEVELS);
    }

    #[test]
    fn affine_uniform_levels_are_evenly_spaced() {
        let gaps: Vec<f32> = AFFINE_UNIFORM_LEVELS
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let first = gaps[0];
        for &gap in &gaps {
            assert!((gap - first).abs() < 1e-6, "gap {gap} != {first}: {gaps:?}");
        }
    }

    #[test]
    fn nf4_shifted_levels_are_not_evenly_spaced() {
        let gaps: Vec<f32> = NF4_SHIFTED_LEVELS.windows(2).map(|w| w[1] - w[0]).collect();
        let first = gaps[0];
        assert!(
            gaps.iter().any(|&gap| (gap - first).abs() > 1e-4),
            "NF4-shifted gaps were all equal: {gaps:?}"
        );
    }

    #[test]
    fn constant_valued_group_round_trips_exactly() {
        // A symmetric 4-bit fit CANNOT do this: every level is scaled from
        // zero, so a nonzero constant always costs some reconstruction
        // error. The affine minimum absorbs the constant exactly.
        let values = [7.5f32; 20];
        let weights = [1.0f32; 20];
        let out = quantize_group_affine(&values, &weights, AffineCodebook::Uniform);
        assert_eq!(out, vec![7.5f32; 20]);
    }

    #[test]
    fn all_zero_group_round_trips_to_exact_zeros() {
        let values = [0.0f32; 10];
        let weights = [1.0f32; 10];
        let out = quantize_group_affine(&values, &weights, AffineCodebook::Nf4Shifted);
        assert_eq!(out, vec![0.0f32; 10]);
    }

    #[test]
    fn near_constant_group_does_not_produce_non_finite_output() {
        let mut values = [1.0f32; 16];
        values[3] += 1e-9; // span well below SPAN_EPS: degenerate path.
        let weights = [1.0f32; 16];
        let out = quantize_group_affine(&values, &weights, AffineCodebook::Uniform);
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn values_on_the_affine_grid_reconstruct_near_exactly() {
        // d0 = 2.0, m0 = 3.0; picks include level 0 and level 15, so
        // min/max land exactly on d0/m0 and the zero-multiplier candidate
        // reproduces them exactly before any refit even runs.
        let levels = AffineCodebook::Uniform.levels();
        let d0 = 2.0f32;
        let m0 = 3.0f32;
        let picks = [0usize, 15, 5, 10, 3, 12];
        let values: Vec<f32> = picks.iter().map(|&i| m0 + d0 * levels[i]).collect();
        let weights = vec![1.0f32; values.len()];
        let out = quantize_group_affine(&values, &weights, AffineCodebook::Uniform);
        for (got, want) in out.iter().zip(values.iter()) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn tail_group_shorter_than_group_size_round_trips_without_error() {
        let values: Vec<f32> = vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.1, 1.0, 1.2, 1.3];
        let weights = vec![1.0f32; values.len()];
        let out = affine_codebook_round_trip(&values, 5, AffineCodebook::Nf4Shifted, &weights);
        assert_eq!(out.len(), values.len());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn affine_error_is_no_worse_than_symmetric_on_a_strong_dc_offset() {
        // The property the whole affine idea rests on: a group sitting far
        // from zero is exactly what a per-group minimum is for.
        let values: Vec<f32> = (0..32).map(|i| 5.0 + 0.05 * i as f32).collect();
        let weights = vec![1.0f32; values.len()];

        let symmetric = quantize_group(&values, &weights, Codebook::Uniform);
        let affine = quantize_group_affine(&values, &weights, AffineCodebook::Uniform);

        let sq_err = |recon: &[f32]| -> f64 {
            values
                .iter()
                .zip(recon)
                .map(|(&x, &r)| {
                    let diff = f64::from(x) - f64::from(r);
                    diff * diff
                })
                .sum()
        };

        assert!(
            sq_err(&affine) <= sq_err(&symmetric),
            "affine err {} > symmetric err {}",
            sq_err(&affine),
            sq_err(&symmetric)
        );
    }
}
