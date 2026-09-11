//! Tests for [`super::two_level_asymmetric`], split into its own file so
//! the logic file stays under this repo's 500-line file limit — the same
//! reason `hats/tcf/tcf-core`'s `asymmetric.rs` keeps
//! `asymmetric_precision_tests.rs` separate.

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
        let (fits, super_d, super_m, pre_divided) = fits_and_super(&values, &weights, precision);
        let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
        let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();
        for ((group, weight_group), fit) in groups.iter().zip(weight_groups.iter()).zip(fits.iter())
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
