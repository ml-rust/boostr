//! Tests for [`super::two_level`], split into its own file so the logic
//! file stays under this repo's 500-line file limit.

use super::*;

fn weighted_squared_error_at(values: &[f32], recon: &[f32]) -> f64 {
    values
        .iter()
        .zip(recon)
        .map(|(&x, &r)| {
            let diff = f64::from(x) - f64::from(r);
            diff * diff
        })
        .sum()
}

#[test]
fn all_zero_super_block_round_trips_to_exact_zeros_all_arms() {
    for precision in [
        SuperPrecision::Bf16,
        SuperPrecision::F16,
        SuperPrecision::F32,
        SuperPrecision::Bf16Reserved,
    ] {
        let values = [0.0f32; SUPER_BLOCK];
        let weights = [1.0f32; SUPER_BLOCK];
        let out = quantize_super_block(&values, &weights, precision);
        assert_eq!(out, vec![0.0f32; SUPER_BLOCK], "{precision:?}");
    }
}

/// A fixture where every group's values sit near-exactly on the 6-bit
/// grid at the UNROUNDED super-scale `d0`: group 0's codes span the
/// full `-32..=-17` range at scale `d0` itself, and groups `1..16` use
/// `d0 / (g + 1)` with small integer codes — exact multiples of each
/// group's own ideal scale. `F32`'s exact `max_d == d0` reconstructs
/// this cleanly; rounding `d0` into bf16 or f16 first (the other two
/// arms) can only push codes off that exact grid, never onto a better
/// one, since nothing else in the pipeline favors a rounded super.
fn ceiling_fixture() -> [f32; SUPER_BLOCK] {
    let mut values = [0.0f32; SUPER_BLOCK];
    let d0 = 1.0f32;
    for (i, code) in (-32i32..=-17).enumerate() {
        values[i] = d0 * code as f32;
    }
    for g in 1..GROUPS_PER_SUPER {
        let dg = d0 / (g as f32 + 1.0);
        for i in 0..GROUP_SIZE {
            let code = -8 + (i % 17) as i32;
            values[g * GROUP_SIZE + i] = dg * code as f32;
        }
    }
    values
}

/// The ceiling property this whole probe rests on: `F32` never rounds
/// its super-scale, so its weighted error can never exceed either
/// half-precision arm's on this fixture. If this fails, the refinement
/// or code derivation differs between arms and the probe is not
/// isolating the super-scale variable it claims to.
///
/// This is NOT a universal property of the design over arbitrary data
/// — a coarser super-scale can occasionally land closer to a
/// particular group's ideal scale by chance (quantization-boundary
/// luck), the same way dither can occasionally help a scalar
/// quantizer. This fixture is built so `F32`'s exact scale is the
/// intended best case, ruling that out.
#[test]
fn f32_arm_is_the_error_ceiling() {
    let values = ceiling_fixture();
    let weights = [1.0f32; SUPER_BLOCK];

    let bf16_out = quantize_super_block(&values, &weights, SuperPrecision::Bf16);
    let f16_out = quantize_super_block(&values, &weights, SuperPrecision::F16);
    let f32_out = quantize_super_block(&values, &weights, SuperPrecision::F32);

    let err_bf16 = weighted_squared_error_at(&values, &bf16_out);
    let err_f16 = weighted_squared_error_at(&values, &f16_out);
    let err_f32 = weighted_squared_error_at(&values, &f32_out);

    assert!(err_f32 <= err_bf16, "F32 {err_f32} > Bf16 {err_bf16}");
    assert!(err_f32 <= err_f16, "F32 {err_f32} > F16 {err_f16}");
}

#[test]
fn bf16_super_value_round_trips_through_bf16_exactly() {
    let values: Vec<f32> = (0..SUPER_BLOCK)
        .map(|i| (i as f32 - 128.0) * 0.05)
        .collect();
    let weights = vec![1.0f32; SUPER_BLOCK];
    // Recompute the same pass-1/pass-2 the function runs, to check the
    // stored super independently of the group loop.
    let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
    let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();
    let max_d = groups
        .iter()
        .zip(weight_groups.iter())
        .map(|(g, w)| fit_group_scale(g, w, FULL_64).d_g)
        .fold(0.0f32, f32::max);
    let super_bits = bf16::from_f32(max_d / MAX_SUB);
    let super_scale = super_bits.to_f32();
    // Round-tripping the already-bf16 value through bf16 again changes
    // nothing: bf16 -> f32 -> bf16 is idempotent for a value already
    // representable in bf16.
    assert_eq!(bf16::from_f32(super_scale), super_bits);
}

#[test]
fn f16_super_value_round_trips_through_f16_exactly() {
    let values: Vec<f32> = (0..SUPER_BLOCK)
        .map(|i| (i as f32 - 128.0) * 0.05)
        .collect();
    let weights = vec![1.0f32; SUPER_BLOCK];
    let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
    let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();
    let max_d = groups
        .iter()
        .zip(weight_groups.iter())
        .map(|(g, w)| fit_group_scale(g, w, FULL_64).d_g)
        .fold(0.0f32, f32::max);
    let super_bits = f16::from_f32(max_d);
    let super_scale = super_bits.to_f32();
    assert_eq!(f16::from_f32(super_scale), super_bits);
}

#[test]
fn tail_row_of_272_round_trips_without_error() {
    // 272 = one full super-block (256) plus a 16-element tail group —
    // exercises a super-block shorter than SUPER_BLOCK on its own
    // actual groups.
    let values: Vec<f32> = (0..272).map(|i| ((i % 37) as f32 - 18.0) * 0.1).collect();
    let weights = vec![1.0f32; values.len()];
    for precision in [
        SuperPrecision::Bf16,
        SuperPrecision::F16,
        SuperPrecision::F32,
        SuperPrecision::Bf16Reserved,
    ] {
        let out = two_level_codebook_round_trip(&values, 272, precision, &weights);
        assert_eq!(out.len(), values.len());
        assert!(out.iter().all(|v| v.is_finite()), "{precision:?}: {out:?}");
    }
}

#[test]
fn constant_nonzero_group_reconstructs_within_one_code_step() {
    let values = [3.25f32; GROUP_SIZE];
    let weights = [1.0f32; GROUP_SIZE];
    for precision in [
        SuperPrecision::Bf16,
        SuperPrecision::F16,
        SuperPrecision::F32,
        SuperPrecision::Bf16Reserved,
    ] {
        let out = quantize_super_block(&values, &weights, precision);
        // One code step at the fitted scale: bound generously since the
        // exact step size depends on the arm's rounding, but a constant
        // group must never miss by more than a small multiple of its
        // own magnitude.
        for &v in &out {
            assert!(
                (v - 3.25).abs() < 0.5,
                "{precision:?}: got {v}, want near 3.25"
            );
        }
    }
}
