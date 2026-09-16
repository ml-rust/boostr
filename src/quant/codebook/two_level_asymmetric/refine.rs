//! Sub-level rounding, refinement, and reconstruction for the asymmetric
//! two-level probe.

use super::fit::asymmetric_reconstruction_error;

/// Highest 6-bit unsigned sub-scale.
pub(super) const MAX_SUB_D: f32 = 63.0;
/// Highest magnitude of the 6-bit signed sub-minimum. The pattern `-32` is
/// reserved (Section 13.4) and never emitted: every clamp below uses
/// `-MAX_SUB_M..=MAX_SUB_M`, not the full `i8` range.
pub(super) const MAX_SUB_M: f32 = 31.0;

/// Highest 4-bit unsigned code, re-exported from `fit` for callers that only
/// need the reconstruction shape, not the fitting search.
pub(super) use super::fit::UMAX;

/// The effective `(d, m)` pair a `(sub_d, sub_m)` choice decodes to, under
/// `precision`'s storage rule: pre-divided ([`super::super::two_level::SuperPrecision::Bf16`])
/// folds the level count into the stored super value, so decode is one
/// multiply per term; not pre-divided (`F16`, `F32`) divides by the level
/// count at decode instead.
pub(super) fn effective_pair(
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
pub(super) fn refine_sub_levels(
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
/// [`super::block::quantize_super_block`] so tests can check the emitted
/// sub-levels directly (in particular, that `sub_m` never lands on the
/// reserved `-32` pattern) without duplicating pass 1 and 2 in the test
/// module.
///
/// The initial rounding, like [`effective_pair`], branches on `pre_divided`:
/// a pre-divided super (`Bf16`) already has the level count folded in, so
/// recovery is `d_g / super_d` directly; a not-pre-divided super (`F16`,
/// `F32`) is the raw extreme, so recovery must multiply back by the level
/// count first — `d_g * MAX_SUB_D / super_d` — exactly mirroring
/// `two_level.rs`'s own `pre_divided` branch for its single super-scale.
#[allow(clippy::too_many_arguments)]
pub(super) fn round_and_refine(
    fit: super::fit::GroupFit,
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
pub(super) fn derive_values(values: &[f32], d_eff: f32, m_eff: f32) -> Vec<f32> {
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
