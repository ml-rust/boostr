//! Two-level super-scale probe, asymmetric variant: mirrors
//! [`super::two_level`]'s super-scale-storage probe on `Q4AS32D_T64`'s
//! geometry instead of `Q6S16D_T64`'s (both retired TCF native encodings) — 4-bit
//! UNSIGNED codes `0..=15`, 32-element groups, 8 groups per 256-element
//! super-block, a 6-bit sub-scale AND a 6-bit signed sub-minimum per group,
//! and ONE super-scale plus ONE super-minimum per super-block.
//!
//! Why this probe exists: `Q6S16D_T64`'s super-scale probe found bf16 worth
//! only a little over f16 at 6.5 bpw. `Q4AS32D_T64` — TCF's best 4-bit
//! encoding, at parity with GGUF's `q4_k` — stores the same bf16-super
//! design TWICE: a super-scale and a super-minimum. This probe isolates
//! whether f16 helps there too, holding every other structural knob (group
//! count, sub-level width, code range) fixed across the three
//! [`SuperPrecision`] arms, exactly as `two_level.rs` does for the
//! symmetric case.
//!
//! [`SuperPrecision::Bf16Reserved`] is not applicable: it retired a reserved
//! CODE on `two_level.rs`'s SIGNED 6-bit grid, and this geometry's codes are
//! UNSIGNED with no reserved pattern. [`two_level_asymmetric_round_trip`]
//! refuses it outright rather than silently mapping it onto [`Bf16`].
//!
//! Mirrors the retired TCF native quantizer's asymmetric super-block
//! search: same three-pass shape (fit every group's `f32` pair, derive the
//! super pair, round and refine each group's sub-levels against it), same
//! weighted least-squares refit closing each candidate, same `±2 x ±2`
//! sub-level search. It differs only where `two_level.rs` already differs:
//! no binary16 rounding of the per-group intermediate (kept exact `f32`),
//! and a local candidate-multiplier sweep in place of that quantizer's
//! search-effort ladder — see [`asymmetric_candidate_multipliers`].
//!
//! [`Bf16`]: SuperPrecision::Bf16

use super::roundtrip::GROUP_SIZE;
use super::two_level::{SuperPrecision, round_bf16, round_f16};
use crate::error::{Error, Result};

/// Groups per super-block: `256 / GROUP_SIZE` with `GROUP_SIZE == 32`,
/// matching `Q4AS32D_T64`'s one super-scale and one super-minimum per 256
/// elements.
const GROUPS_PER_SUPER: usize = 8;
/// Elements per super-block.
const SUPER_BLOCK: usize = GROUP_SIZE * GROUPS_PER_SUPER;

/// Highest 4-bit unsigned code.
const UMAX: f32 = 15.0;
/// Highest 6-bit unsigned sub-scale.
const MAX_SUB_D: f32 = 63.0;
/// Highest magnitude of the 6-bit signed sub-minimum. The pattern `-32` is
/// reserved (Section 13.4) and never emitted: every clamp below uses
/// `-MAX_SUB_M..=MAX_SUB_M`, not the full `i8` range.
const MAX_SUB_M: f32 = 31.0;

/// One group's ideal (pre-super-rounding) float pair `(d_g, m_g)`.
#[derive(Clone, Copy)]
struct GroupFit {
    d_g: f32,
    m_g: f32,
}

/// The asymmetric pair search's candidate multipliers, mirroring the SHAPE
/// of the retired TCF native quantizer's standard effort: one-sided,
/// `0.1 * i` for `i` in `0..=20` (21 candidates), on top of the
/// unconditional candidate-0 min/max fit tried separately. That quantizer
/// kept the list behind a search-effort enum, so this is a new
/// local sweep, not a duplicate of this module's own (differently shaped,
/// two-sided) [`super::quantize::candidate_multipliers`].
fn asymmetric_candidate_multipliers() -> impl Iterator<Item = f32> {
    (0..=20).map(|i| 0.1 * i as f32)
}

/// Weighted sum of squared reconstruction error
/// `sum(w_i * (x_i - (d * u_i + m))^2)` where
/// `u_i = clamp(RN_even((x_i - m) / d), 0, UMAX)`. `d` MUST be finite and
/// strictly positive — both callers already guarantee this before scoring;
/// [`refine_sub_levels`] handles its own `d_eff == 0.0` case separately, the
/// same way [`super::two_level`]'s `refine_sub_scale` does for the
/// symmetric arm.
fn asymmetric_reconstruction_error(values: &[f32], weights: &[f32], d: f32, m: f32) -> f64 {
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
fn fit_group_pair(values: &[f32], weights: &[f32]) -> GroupFit {
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

/// The effective `(d, m)` pair a `(sub_d, sub_m)` choice decodes to, under
/// `precision`'s storage rule: pre-divided ([`SuperPrecision::Bf16`]) folds
/// the level count into the stored super value, so decode is one multiply
/// per term; not pre-divided ([`SuperPrecision::F16`], [`SuperPrecision::F32`])
/// divides by the level count at decode instead.
fn effective_pair(
    super_d: f32,
    super_m: f32,
    sub_d: u8,
    sub_m: i8,
    pre_divided: bool,
) -> (f32, f32) {
    if pre_divided {
        (super_d * f32::from(sub_d), super_m * f32::from(sub_m))
    } else {
        (
            super_d * f32::from(sub_d) / MAX_SUB_D,
            super_m * f32::from(sub_m) / MAX_SUB_M,
        )
    }
}

/// Refines a group's rounded `(sub_d, sub_m)` guess over the `±2 x ±2`
/// integer neighborhood (25 pairs), mirroring the retired native
/// quantizer's sub-level refinement: each candidate is scored by its OWN
/// effective pair via [`effective_pair`], first-on-tie, with the rounded
/// pair always among the candidates so the search never scores worse than
/// plain rounding.
///
/// `sub_d` is clamped to `[0, 63]` and `sub_m` to `[-31, 31]`, so the
/// reserved sub-minimum pattern `-32` is unreachable here exactly as it is
/// in the rounding this refines. A super-minimum of zero pins `sub_m` to 0:
/// every level would decode to the same zero minimum, so searching around a
/// nonzero `rounded_m` would explore a difference the wire cannot carry.
/// Duplicate neighbors produced by clamping at either end of a range are
/// skipped, matching that quantizer's own dedup.
fn refine_sub_levels(
    values: &[f32],
    weights: &[f32],
    rounded_d: u8,
    rounded_m: i8,
    super_d: f32,
    super_m: f32,
    pre_divided: bool,
) -> (u8, i8) {
    let mut best: Option<(f64, u8, i8)> = None;
    let mut previous_d: Option<u8> = None;
    for dd in -2i32..=2 {
        let level_d = i32::from(rounded_d)
            .saturating_add(dd)
            .clamp(0, MAX_SUB_D as i32);
        let Ok(sub_d) = u8::try_from(level_d) else {
            continue;
        };
        if previous_d == Some(sub_d) {
            continue;
        }
        previous_d = Some(sub_d);

        let mut previous_m: Option<i8> = None;
        for dm in -2i32..=2 {
            let level_m = if super_m > 0.0 {
                i32::from(rounded_m)
                    .saturating_add(dm)
                    .clamp(-(MAX_SUB_M as i32), MAX_SUB_M as i32)
            } else {
                0
            };
            let Ok(sub_m) = i8::try_from(level_m) else {
                continue;
            };
            if previous_m == Some(sub_m) {
                continue;
            }
            previous_m = Some(sub_m);

            let (d_eff, m_eff) = effective_pair(super_d, super_m, sub_d, sub_m, pre_divided);
            if !d_eff.is_finite() || d_eff < 0.0 || !m_eff.is_finite() {
                continue;
            }
            // A zero effective scale forces every code to 0 (see
            // `derive_values`); scoring it needs the raw sum-of-squares
            // against the constant `m_eff`, not a division by a zero scale.
            let err: f64 = if d_eff == 0.0 {
                values
                    .iter()
                    .enumerate()
                    .map(|(index, &x)| {
                        let w = f64::from(weights.get(index).copied().unwrap_or(1.0));
                        let diff = f64::from(x) - f64::from(m_eff);
                        w * diff * diff
                    })
                    .sum()
            } else {
                asymmetric_reconstruction_error(values, weights, d_eff, m_eff)
            };
            let improves = match best {
                None => true,
                Some((best_err, _, _)) => err < best_err,
            };
            if improves {
                best = Some((err, sub_d, sub_m));
            }
        }
    }
    match best {
        Some((_, sub_d, sub_m)) => (sub_d, sub_m),
        None => (rounded_d, rounded_m),
    }
}

/// Rounds one group's fitted pair against the super pair, then refines the
/// rounded guess over `±2 x ±2` — pass 3's per-group work, factored out of
/// [`quantize_super_block`] so tests can check the emitted sub-levels
/// directly (in particular, that `sub_m` never lands on the reserved `-32`
/// pattern) without duplicating pass 1 and 2 in the test module.
///
/// The initial rounding, like [`effective_pair`], branches on `pre_divided`:
/// a pre-divided super (`Bf16`) already has the level count folded in, so
/// recovery is `d_g / super_d` directly; a not-pre-divided super (`F16`,
/// `F32`) is the raw extreme, so recovery must multiply back by the level
/// count first — `d_g * MAX_SUB_D / super_d` — exactly mirroring
/// `two_level.rs`'s own `pre_divided` branch for its single super-scale.
fn round_and_refine(
    fit: GroupFit,
    super_d: f32,
    super_m: f32,
    pre_divided: bool,
    group: &[f32],
    weight_group: &[f32],
) -> (u8, i8) {
    let raw_d = if super_d > 0.0 {
        if pre_divided {
            fit.d_g / super_d
        } else {
            fit.d_g * MAX_SUB_D / super_d
        }
    } else {
        0.0
    };
    let rounded_d = raw_d.round_ties_even().clamp(0.0, MAX_SUB_D) as u8;
    let raw_m = if super_m > 0.0 {
        if pre_divided {
            fit.m_g / super_m
        } else {
            fit.m_g * MAX_SUB_M / super_m
        }
    } else {
        0.0
    };
    let rounded_m = raw_m.round_ties_even().clamp(-MAX_SUB_M, MAX_SUB_M) as i8;
    refine_sub_levels(
        group,
        weight_group,
        rounded_d,
        rounded_m,
        super_d,
        super_m,
        pre_divided,
    )
}

/// Reconstructs a group at effective pair `(d_eff, m_eff)`:
/// `u = clamp(RN_even((x - m_eff) / d_eff), 0, UMAX)`, `x' = m_eff + d_eff *
/// u`. `d_eff == 0.0` reconstructs every element to the constant `m_eff` —
/// the zero-scale form, where the minimum alone carries the group's value,
/// covering both the all-zero case (`m_eff == 0.0` too) and a nonzero
/// constant group.
fn derive_values(values: &[f32], d_eff: f32, m_eff: f32) -> Vec<f32> {
    if d_eff == 0.0 {
        return vec![m_eff; values.len()];
    }
    values
        .iter()
        .map(|&x| {
            let u = ((x - m_eff) / d_eff).round_ties_even().clamp(0.0, UMAX);
            m_eff + d_eff * u
        })
        .collect()
}

/// Derives `(super_d, super_m, pre_divided)` from a super-block's group
/// fits, per `precision`'s storage rule. Returns `None` for
/// [`SuperPrecision::Bf16Reserved`] (unreachable through the public entry
/// point, which refuses that arm first) and for a non-finite or negative
/// rounding result, both of which [`quantize_super_block`] treats as the
/// exact-zero fallback rather than propagate further.
fn super_pair(precision: SuperPrecision, max_d: f32, max_abs_m: f32) -> Option<(f32, f32, bool)> {
    let (super_d, super_m, pre_divided) = match precision {
        SuperPrecision::Bf16 => (
            round_bf16(max_d / MAX_SUB_D),
            round_bf16(max_abs_m / MAX_SUB_M),
            true,
        ),
        SuperPrecision::F16 => (round_f16(max_d), round_f16(max_abs_m), false),
        SuperPrecision::F32 => (max_d, max_abs_m, false),
        SuperPrecision::Bf16Reserved => return None,
    };
    if !super_d.is_finite() || super_d < 0.0 || !super_m.is_finite() || super_m < 0.0 {
        return None;
    }
    Some((super_d, super_m, pre_divided))
}

/// Quantizes then dequantizes one super-block (up to [`SUPER_BLOCK`]
/// elements, grouped into up to [`GROUPS_PER_SUPER`] groups of
/// [`GROUP_SIZE`]) under `precision`.
///
/// Three passes, mirroring `Q4AS32D_T64`'s own three-pass structure:
/// 1. [`fit_group_pair`] every group's ideal `f32` pair independently.
/// 2. Derive ONE super-scale from `max(d_g)` and ONE super-minimum from
///    `max(|m_g|)` across the block, per `precision`'s storage rule (see
///    [`super_pair`]).
/// 3. Per group, [`round_and_refine`] then [`derive_values`] against the
///    FINAL effective pair.
///
/// A super-block where every group is all-zero (`max(d_g) == 0.0` AND
/// `max(|m_g|) == 0.0`) stores both supers as `0.0` and every group
/// reconstructs to exact zeros without running steps 2-3's rounding, which
/// would otherwise divide by zero. A super-block where every group is
/// CONSTANT but not all zero (`max(d_g) == 0.0`, `max(|m_g|) > 0.0`) is not
/// this case: `super_d` alone is `0.0`, `sub_d` is pinned to `0` per group,
/// and the nonzero constants still round-trip exactly through `super_m`.
///
/// [`SuperPrecision::Bf16Reserved`] never reaches this function through the
/// public entry point ([`two_level_asymmetric_round_trip`] refuses it
/// first); [`super_pair`] returning `None` for it here is a second,
/// defensive line rather than the primary refusal.
fn quantize_super_block(values: &[f32], weights: &[f32], precision: SuperPrecision) -> Vec<f32> {
    let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
    let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();

    // Pass 1: every group's ideal float pair.
    let fits: Vec<GroupFit> = groups
        .iter()
        .zip(weight_groups.iter())
        .map(|(g, w)| fit_group_pair(g, w))
        .collect();
    let max_d = fits.iter().fold(0.0f32, |acc, f| acc.max(f.d_g));
    let max_abs_m = fits.iter().fold(0.0f32, |acc, f| acc.max(f.m_g.abs()));

    if max_d == 0.0 && max_abs_m == 0.0 {
        return vec![0.0f32; values.len()];
    }

    // Pass 2: the super pair, per arm.
    let Some((super_d, super_m, pre_divided)) = super_pair(precision, max_d, max_abs_m) else {
        // Degenerate rounding, or the unreachable Bf16Reserved arm: fall
        // back to exact zeros rather than propagate a missing pair into
        // every group's codes.
        return vec![0.0f32; values.len()];
    };

    // Pass 3: per group, sub-levels then values against the final effective
    // pair.
    let mut out = Vec::with_capacity(values.len());
    for ((group, weight_group), fit) in groups.iter().zip(weight_groups.iter()).zip(fits.iter()) {
        let (sub_d, sub_m) =
            round_and_refine(*fit, super_d, super_m, pre_divided, group, weight_group);
        let (d_eff, m_eff) = effective_pair(super_d, super_m, sub_d, sub_m, pre_divided);
        out.extend(derive_values(group, d_eff, m_eff));
    }
    out
}

/// Quantizes then dequantizes every value in `values` against `precision`,
/// grouping [`SUPER_BLOCK`] consecutive elements per row exactly like
/// [`super::two_level::two_level_codebook_round_trip`] — same tail handling
/// (a super-block shorter than [`SUPER_BLOCK`] is quantized on its own
/// actual groups, never padded), same never-crosses-a-row-boundary rule,
/// same `in_features == 0` no-op.
///
/// # Errors
/// Returns [`Error::QuantError`] if `precision` is
/// [`SuperPrecision::Bf16Reserved`]: that arm retired a reserved CODE on the
/// symmetric probe's signed grid, and this geometry's codes are unsigned
/// with no reserved pattern to retire, so there is no faithful meaning to
/// map it onto.
pub fn two_level_asymmetric_round_trip(
    values: &[f32],
    in_features: usize,
    precision: SuperPrecision,
    weights: &[f32],
) -> Result<Vec<f32>> {
    if precision == SuperPrecision::Bf16Reserved {
        return Err(Error::QuantError {
            reason: "two_level_asymmetric_round_trip: Bf16Reserved is not applicable to the \
                     asymmetric arm — its unsigned codes have no reserved pattern to retire, \
                     unlike the symmetric probe's signed grid"
                .to_string(),
        });
    }
    if in_features == 0 {
        return Ok(values.to_vec());
    }
    let mut out = Vec::with_capacity(values.len());
    let value_rows = values.chunks(in_features);
    let weight_rows = weights.chunks(in_features);
    for (row, weight_row) in value_rows.zip(weight_rows) {
        for (super_block, weight_super_block) in
            row.chunks(SUPER_BLOCK).zip(weight_row.chunks(SUPER_BLOCK))
        {
            out.extend(quantize_super_block(
                super_block,
                weight_super_block,
                precision,
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "two_level_asymmetric_tests.rs"]
mod tests;
