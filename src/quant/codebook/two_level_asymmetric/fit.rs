//! Per-group `(d, m)` pair fitting for the asymmetric two-level probe.

/// Highest 4-bit unsigned code.
pub(super) const UMAX: f32 = 15.0;

/// One group's ideal (pre-super-rounding) float pair `(d_g, m_g)`.
#[derive(Clone, Copy)]
pub(super) struct GroupFit {
    pub(super) d_g: f32,
    pub(super) m_g: f32,
}

/// The asymmetric pair search's candidate multipliers, mirroring the SHAPE
/// of the retired TCF native quantizer's standard effort: one-sided,
/// `0.1 * i` for `i` in `0..=20` (21 candidates), on top of the
/// unconditional candidate-0 min/max fit tried separately. That quantizer
/// kept the list behind a search-effort enum, so this is a new
/// local sweep, not a duplicate of this module's own (differently shaped,
/// two-sided) [`super::super::quantize::candidate_multipliers`].
pub(super) fn asymmetric_candidate_multipliers() -> impl Iterator<Item = f32> {
    (0..=20).map(|i| 0.1 * i as f32)
}

/// Weighted sum of squared reconstruction error
/// `sum(w_i * (x_i - (d * u_i + m))^2)` where
/// `u_i = clamp(RN_even((x_i - m) / d), 0, UMAX)`. `d` MUST be finite and
/// strictly positive — both callers already guarantee this before scoring;
/// `refine::refine_sub_levels` handles its own `d_eff == 0.0` case
/// separately, the same way `two_level::fit`'s `refine_sub_scale` does for
/// the symmetric arm.
pub(super) fn asymmetric_reconstruction_error(
    values: &[f32],
    weights: &[f32],
    d: f32,
    m: f32,
) -> f64 {
    let mut err = 0.0f64;
    for (index, &x) in values.iter().enumerate() {
        let u = ((x - m) / d).round_ties_even().clamp(0.0, UMAX);
        let diff = f64::from(x) - (f64::from(d) * f64::from(u) + f64::from(m));
        let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
        err += w * diff * diff;
    }
    err
}

/// Fits one group's ideal `f32` pair `(d_g, m_g)`, mirroring the retired
/// native quantizer's group fit: candidate 0 is always the plain min/max fit
/// (`d = (hi - lo) / UMAX`, `m = lo`); every remaining candidate takes
/// `inv = (-1 + multiplier + UMAX) / (hi - lo)` from
/// [`asymmetric_candidate_multipliers`], rounds provisional codes, and
/// closes with the weighted least-squares refit of `(d, m)` against those
/// codes. Every candidate is scored by [`asymmetric_reconstruction_error`]
/// against the value it actually reconstructs; first-on-tie (a strictly
/// lower error is required to replace the incumbent), matching this
/// module's other searches.
///
/// A degenerate group (`values` empty, or every value equal) is not
/// searched: an empty group has no pair to fit, and a constant group's exact
/// answer is `d_g = 0.0`, `m_g` = the constant itself, no search needed.
/// `d_g == 0.0` here is NOT "all zero" the way it is in the symmetric arm —
/// it is "constant", and a nonzero constant's value lives entirely in
/// `m_g`. This is the property asymmetric geometry buys over symmetric.
pub(super) fn fit_group_pair(values: &[f32], weights: &[f32]) -> GroupFit {
    if values.is_empty() {
        return GroupFit { d_g: 0.0, m_g: 0.0 };
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in values {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if hi == lo {
        return GroupFit { d_g: 0.0, m_g: lo };
    }

    // Candidate 0: the plain min/max fit, tried before the sweep so the
    // search is always at least this good. `hi != lo` and `UMAX > 0` make
    // `d0` finite and strictly positive unconditionally, so `best` always
    // starts seeded.
    let d0 = (hi - lo) / UMAX;
    let err0 = asymmetric_reconstruction_error(values, weights, d0, lo);
    let mut best: (f64, f32, f32) = (err0, d0, lo);

    for multiplier in asymmetric_candidate_multipliers() {
        let inv = (-1.0 + multiplier + UMAX) / (hi - lo);
        let mut n = 0.0f64;
        let mut sum_l = 0.0f64;
        let mut sum_l2 = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_xl = 0.0f64;
        for (index, &x) in values.iter().enumerate() {
            let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
            let l = f64::from((inv * (x - lo)).round_ties_even().clamp(0.0, UMAX));
            n += w;
            sum_l += w * l;
            sum_l2 += w * (l * l);
            sum_x += w * f64::from(x);
            sum_xl += w * (f64::from(x) * l);
        }
        let det = n * sum_l2 - sum_l * sum_l;
        if det <= 0.0 {
            continue;
        }
        let d = ((n * sum_xl - sum_x * sum_l) / det) as f32;
        let m = ((sum_l2 * sum_x - sum_l * sum_xl) / det) as f32;
        if d <= 0.0 || !d.is_finite() || !m.is_finite() {
            continue;
        }
        let err = asymmetric_reconstruction_error(values, weights, d, m);
        if err < best.0 {
            best = (err, d, m);
        }
    }

    GroupFit {
        d_g: best.1,
        m_g: best.2,
    }
}
