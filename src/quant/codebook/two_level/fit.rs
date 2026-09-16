//! Per-group scale fitting and refinement shared by every [`super::SuperPrecision`]
//! arm: same group fit, same sub-scale search, same code derivation — the
//! arms differ only in how the super-scale itself is stored (`super::block`).

use super::super::quantize::candidate_multipliers;

/// The signed 6-bit code grid an arm quantizes onto. Two grids exist because
/// TCF's retired native `Q6S16D_T64` USED TO reserve the most-negative
/// code as a rejection point (retired: no CODE plane reserves a value now,
/// full two's-complement range like ggml's Q6_K); the two differ by one
/// level in 64, and isolating that cost is what motivated the retirement —
/// [`super::SuperPrecision::Bf16Reserved`] keeps the old grid around as the
/// historical comparison point.
#[derive(Debug, Clone, Copy)]
pub(super) struct CodeGrid {
    /// The divisor the fit sweep anchors on: `d = max_abs / (qmax + mult)`.
    pub(super) qmax: f32,
    /// Lowest code emitted.
    pub(super) lo: f32,
    /// Highest code emitted.
    pub(super) hi: f32,
}

/// All 64 codes, `-32..=31`. Q6_K's grid, and TCF's current grid for every
/// symmetric encoding since Section 13.2's reservation was retired
/// (`geometry.qmax()` is still `2^(bits-1) - 1`; the lower bound widened to
/// `-(qmax + 1)`).
pub(super) const FULL_64: CodeGrid = CodeGrid {
    qmax: 32.0,
    lo: -32.0,
    hi: 31.0,
};

/// 63 codes, `-31..=31`, the most-negative pattern reserved. TCF's grid for
/// every symmetric encoding BEFORE Section 13.2's reservation was retired;
/// kept only as [`super::SuperPrecision::Bf16Reserved`]'s historical
/// comparison point.
pub(super) const RESERVED_63: CodeGrid = CodeGrid {
    qmax: 31.0,
    lo: -31.0,
    hi: 31.0,
};

/// One group's ideal (pre-super-rounding) float scale `d_g`, kept for the
/// super-block pass that follows.
pub(super) struct GroupFit {
    pub(super) d_g: f32,
}

/// Nearest integer code to `u`, round-to-nearest-even, clamped to the
/// grid. The uniform-integer analogue of
/// [`super::super::quantize::nearest_level`]'s table lookup: there is no
/// 16-entry table here, so the "nearest level" is just an integer
/// round-and-clamp.
pub(super) fn nearest_code_f(u: f32, grid: CodeGrid) -> f32 {
    u.round_ties_even().clamp(grid.lo, grid.hi)
}

/// Weighted squared error of reconstructing `values` at scale `d`, codes
/// taken to the nearest 6-bit integer. Mirrors `quantize.rs`'s own
/// `weighted_squared_error` with a uniform-integer level set instead of a
/// table.
pub(super) fn weighted_squared_error_uniform(
    values: &[f32],
    weights: &[f32],
    d: f32,
    grid: CodeGrid,
) -> f64 {
    let mut err = 0.0f64;
    for (index, &x) in values.iter().enumerate() {
        let code = nearest_code_f(x / d, grid);
        let diff = f64::from(x) - f64::from(d) * f64::from(code);
        let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
        err += w * diff * diff;
    }
    err
}

/// Fits one group's ideal `f32` scale `d_g` by the same 19-candidate
/// weighted sweep the other codebook probes use: `d = max_abs / (QMAX +
/// multiplier)`, scored by weighted squared error, first-on-tie (strictly
/// lower error required to replace the incumbent). `d_g` is kept as `f32`
/// — NOT rounded — since it is only pass 1's input to the super-block's
/// `max(d_g)`; rounding here would just discard precision before pass 2
/// gets to choose the storage format.
///
/// An all-zero or degenerate group (every candidate skipped) returns
/// `d_g = 0.0`, the exact-zero form: no live scale to search over.
pub(super) fn fit_group_scale(values: &[f32], weights: &[f32], grid: CodeGrid) -> GroupFit {
    let max_abs = values.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return GroupFit { d_g: 0.0 };
    }
    let mut best: Option<(f64, f32)> = None;
    for multiplier in candidate_multipliers() {
        let d = max_abs / (grid.qmax + multiplier);
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        let err = weighted_squared_error_uniform(values, weights, d, grid);
        let improves = match best {
            None => true,
            Some((best_err, _)) => err < best_err,
        };
        if improves {
            best = Some((err, d));
        }
    }
    match best {
        Some((_, d_g)) => GroupFit { d_g },
        None => GroupFit { d_g: 0.0 },
    }
}

/// Refines an initial `u8` sub-scale guess over `±2` (the same radius the
/// retired `Q6S16D_T64` quantizer applied), scoring each neighbor by weighted squared
/// error against ITS OWN effective scale (`effective(sub)`), keeping the
/// strictly-lower-error candidate so the search never moves off the
/// rounded start without cause.
pub(super) fn refine_sub_scale(
    values: &[f32],
    weights: &[f32],
    rounded: u8,
    grid: CodeGrid,
    max_sub: f32,
    effective: impl Fn(u8) -> f32,
) -> u8 {
    let mut best: Option<(f64, u8)> = None;
    for delta in -2i32..=2 {
        let candidate = i32::from(rounded)
            .saturating_add(delta)
            .clamp(0, max_sub as i32);
        let Ok(sub) = u8::try_from(candidate) else {
            continue;
        };
        let eff = effective(sub);
        if !eff.is_finite() || eff < 0.0 {
            continue;
        }
        // A zero effective scale forces every code to 0 (see
        // `derive_codes`); scoring it needs the raw sum-of-squares, not a
        // division by a zero scale.
        let err: f64 = if eff == 0.0 {
            values
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let w = f64::from(weights.get(i).copied().unwrap_or(1.0));
                    w * f64::from(x) * f64::from(x)
                })
                .sum()
        } else {
            weighted_squared_error_uniform(values, weights, eff, grid)
        };
        let improves = match best {
            None => true,
            Some((best_err, _)) => err < best_err,
        };
        if improves {
            best = Some((err, sub));
        }
    }
    best.map(|(_, sub)| sub).unwrap_or(rounded)
}

/// Reconstructs a group at effective scale `effective`: `round_ties_even`,
/// clamp to `[-32, 31]`. `effective == 0.0` reconstructs exact zeros — the
/// group's live scale rounded away entirely, matching the all-zero rule
/// every codebook probe in this module applies.
pub(super) fn derive_codes(values: &[f32], effective: f32, grid: CodeGrid) -> Vec<f32> {
    if effective == 0.0 {
        return vec![0.0f32; values.len()];
    }
    values
        .iter()
        .map(|&x| effective * nearest_code_f(x / effective, grid))
        .collect()
}
