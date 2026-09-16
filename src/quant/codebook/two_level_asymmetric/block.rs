//! Super-block quantization for the asymmetric two-level probe: derives ONE
//! super-scale and ONE super-minimum per block from the per-group fits in
//! [`super::fit`], per [`SuperPrecision`]'s storage rule, then re-derives
//! every group's sub-levels via [`super::refine`]. This is the module's
//! public entry point.

use super::super::roundtrip::GROUP_SIZE;
use super::super::two_level::{SuperPrecision, round_bf16, round_f16};
use super::fit::{GroupFit, fit_group_pair};
use super::refine::{MAX_SUB_D, MAX_SUB_M, derive_values, effective_pair, round_and_refine};
use crate::error::{Error, Result};

/// Groups per super-block: `256 / GROUP_SIZE` with `GROUP_SIZE == 32`,
/// matching `Q4AS32D_T64`'s one super-scale and one super-minimum per 256
/// elements.
const GROUPS_PER_SUPER: usize = 8;
/// Elements per super-block.
const SUPER_BLOCK: usize = GROUP_SIZE * GROUPS_PER_SUPER;

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
/// `GROUP_SIZE`) under `precision`.
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
/// [`super::super::two_level::two_level_codebook_round_trip`] — same tail
/// handling (a super-block shorter than [`SUPER_BLOCK`] is quantized on its
/// own actual groups, never padded), same never-crosses-a-row-boundary
/// rule, same `in_features == 0` no-op.
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
mod tests {
    use super::*;
    use half::{bf16, f16};

    const ARMS: [SuperPrecision; 3] = [
        SuperPrecision::Bf16,
        SuperPrecision::F16,
        SuperPrecision::F32,
    ];

    /// Recomputes pass 1's fits and pass 2's super pair independently of
    /// [`quantize_super_block`], for tests that need to inspect either.
    fn fits_and_super(
        values: &[f32],
        weights: &[f32],
        precision: SuperPrecision,
    ) -> (Vec<GroupFit>, f32, f32, bool) {
        let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
        let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();
        let fits: Vec<GroupFit> = groups
            .iter()
            .zip(weight_groups.iter())
            .map(|(g, w)| fit_group_pair(g, w))
            .collect();
        let max_d = fits.iter().fold(0.0f32, |acc, f| acc.max(f.d_g));
        let max_abs_m = fits.iter().fold(0.0f32, |acc, f| acc.max(f.m_g.abs()));
        let (super_d, super_m, pre_divided) =
            super_pair(precision, max_d, max_abs_m).unwrap_or((0.0, 0.0, false));
        (fits, super_d, super_m, pre_divided)
    }

    #[test]
    fn all_zero_super_block_round_trips_to_exact_zeros_all_arms() {
        for precision in ARMS {
            let values = [0.0f32; SUPER_BLOCK];
            let weights = [1.0f32; SUPER_BLOCK];
            let out = quantize_super_block(&values, &weights, precision);
            assert_eq!(out, vec![0.0f32; SUPER_BLOCK], "{precision:?}");
        }
    }

    /// The property asymmetric geometry buys over symmetric: a constant
    /// NONZERO group reconstructs EXACTLY, because the minimum alone carries it
    /// (`d_eff == 0.0`, every code 0, `x' = m_eff`).
    ///
    /// The constant is `31.0`, not an arbitrary value: it makes every arm's
    /// round trip land on an EXACT floating-point identity rather than merely a
    /// close one, so this test demonstrates the guarantee rather than getting
    /// lucky on it. With one group in the block, `max_abs_m == 31.0`.
    /// `Bf16`: `super_m = bf16(31.0 / 31.0) = bf16(1.0) = 1.0` exactly (`1.0` has
    /// a trivial bf16 mantissa), so `m_eff = super_m * 31 = 31.0` exactly.
    /// `F16`/`F32`: `super_m = 31.0` exactly (small integers are exact in both),
    /// and `31.0 * 31.0 = 961.0` is exactly representable in `f32`, so
    /// `961.0 / 31.0 = 31.0` exactly at every step of both the sub-minimum
    /// rounding and the decode. A generic non-integer constant does not carry
    /// this guarantee under `Bf16`'s pre-divided rounding — see `two_level.rs`'s
    /// own super-scale tests, which bound rather than assert exact equality for
    /// exactly that reason.
    #[test]
    fn constant_nonzero_group_reconstructs_exactly() {
        for precision in ARMS {
            let values = [31.0f32; GROUP_SIZE];
            let weights = [1.0f32; GROUP_SIZE];
            let out = quantize_super_block(&values, &weights, precision);
            for &v in &out {
                assert_eq!(v, 31.0, "{precision:?}: got {v}, want exactly 31.0");
            }
        }
    }

    #[test]
    fn bf16_super_values_round_trip_through_bf16_exactly() {
        let values: Vec<f32> = (0..SUPER_BLOCK)
            .map(|i| (i as f32 - 128.0) * 0.05 + 10.0)
            .collect();
        let weights = vec![1.0f32; SUPER_BLOCK];
        let (fits, ..) = fits_and_super(&values, &weights, SuperPrecision::Bf16);
        let max_d = fits.iter().fold(0.0f32, |acc, f| acc.max(f.d_g));
        let max_abs_m = fits.iter().fold(0.0f32, |acc, f| acc.max(f.m_g.abs()));

        let super_d_bits = bf16::from_f32(max_d / MAX_SUB_D);
        let super_m_bits = bf16::from_f32(max_abs_m / MAX_SUB_M);
        assert_eq!(bf16::from_f32(super_d_bits.to_f32()), super_d_bits);
        assert_eq!(bf16::from_f32(super_m_bits.to_f32()), super_m_bits);
    }

    #[test]
    fn f16_super_values_round_trip_through_f16_exactly() {
        let values: Vec<f32> = (0..SUPER_BLOCK)
            .map(|i| (i as f32 - 128.0) * 0.05 + 10.0)
            .collect();
        let weights = vec![1.0f32; SUPER_BLOCK];
        let (fits, ..) = fits_and_super(&values, &weights, SuperPrecision::F16);
        let max_d = fits.iter().fold(0.0f32, |acc, f| acc.max(f.d_g));
        let max_abs_m = fits.iter().fold(0.0f32, |acc, f| acc.max(f.m_g.abs()));

        let super_d_bits = f16::from_f32(max_d);
        let super_m_bits = f16::from_f32(max_abs_m);
        assert_eq!(f16::from_f32(super_d_bits.to_f32()), super_d_bits);
        assert_eq!(f16::from_f32(super_m_bits.to_f32()), super_m_bits);
    }

    #[test]
    fn sub_minimum_never_emits_reserved_negative_32() {
        // Every group's ideal minimum sits far more negative than any
        // super-minimum this block can express, forcing the rounding toward
        // the reserved boundary from below.
        let mut values = [0.0f32; SUPER_BLOCK];
        for (g, group) in values.chunks_mut(GROUP_SIZE).enumerate() {
            let base = -1000.0 * (g as f32 + 1.0);
            for (i, v) in group.iter_mut().enumerate() {
                *v = base + i as f32;
            }
        }
        let weights = [1.0f32; SUPER_BLOCK];
        for precision in ARMS {
            let (fits, super_d, super_m, pre_divided) =
                fits_and_super(&values, &weights, precision);
            let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
            let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();
            for ((group, weight_group), fit) in
                groups.iter().zip(weight_groups.iter()).zip(fits.iter())
            {
                let (_, sub_m) =
                    round_and_refine(*fit, super_d, super_m, pre_divided, group, weight_group);
                assert!(
                    sub_m > -32,
                    "{precision:?}: sub_m {sub_m} hit the reserved pattern"
                );
                assert!(
                    (-31..=31).contains(&sub_m),
                    "{precision:?}: sub_m {sub_m} out of range"
                );
            }
        }
    }

    #[test]
    fn tail_row_of_272_round_trips_without_error() {
        // 272 = one full super-block (256) plus a 16-element tail group —
        // exercises a super-block shorter than SUPER_BLOCK on its own actual
        // groups.
        let values: Vec<f32> = (0..272).map(|i| ((i % 37) as f32 - 18.0) * 0.1).collect();
        let weights = vec![1.0f32; values.len()];
        for precision in ARMS {
            let result = two_level_asymmetric_round_trip(&values, 272, precision, &weights);
            assert!(result.is_ok(), "{precision:?}: unexpected error");
            if let Ok(out) = result {
                assert_eq!(out.len(), values.len());
                assert!(out.iter().all(|v| v.is_finite()), "{precision:?}: {out:?}");
            }
        }
    }

    #[test]
    fn bf16_reserved_is_refused_by_the_entry_point() {
        let values = [1.0f32; GROUP_SIZE];
        let weights = [1.0f32; GROUP_SIZE];
        let result = two_level_asymmetric_round_trip(
            &values,
            GROUP_SIZE,
            SuperPrecision::Bf16Reserved,
            &weights,
        );
        assert!(
            result.is_err(),
            "Bf16Reserved must be refused, not silently mapped"
        );
    }
}
