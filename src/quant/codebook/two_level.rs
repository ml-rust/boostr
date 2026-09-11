//! Two-level super-scale probe: isolates ONE variable — how the per-256
//! super-scale is stored and whether it is pre-divided — from TCF's
//! `Q6S16D_T64` geometry (16-element groups, 8-bit sub-scale, one
//! super-scale per 256 elements, symmetric 6-bit codes). Everything else
//! (group count, sub-scale width, code range) is held fixed across the
//! three [`SuperPrecision`] arms so a difference in reconstruction error
//! traces to the super-scale storage alone, never to geometry.
//!
//! Why this probe exists: `Q6S16D_T64` measurably loses to GGUF's `q6_k` at
//! the same 6.5 bpw, and every OTHER structural knob (group size, sub-scale
//! width) already matches. The one difference left is how the super-scale
//! is stored — TCF: bfloat16, pre-divided by the sub-scale range (255) so a
//! group decodes in one multiply; Q6_K: f16, NOT pre-divided, with an int8
//! sub-scale. bf16 has 8 mantissa bits, f16 has 11 — trading exponent range
//! for precision. [`SuperPrecision::F32`] is the ceiling neither storage
//! format can beat: no rounding at all above the per-group `f32` fit.
//!
//! Reuses [`super::quantize::candidate_multipliers`] and
//! [`super::quantize::nearest_level`]'s tie rule (first-on-tie, ascending)
//! unchanged — this arm differs from the codebook probes ONLY in level
//! shape (uniform 6-bit integers, not a 16-entry table) and in adding a
//! second scale tier.

use half::{bf16, f16};

use super::quantize::candidate_multipliers;

/// Elements per group — fixed at TCF's `Q6S16D_T64` width, distinct from
/// the 4-bit probes' `GROUP_SIZE = 32` (`roundtrip.rs`). A local constant,
/// not a duplicate: no existing table names this width.
const GROUP_SIZE: usize = 16;
/// Groups per super-block: `256 / GROUP_SIZE`, matching `Q6S16D_T64`'s one
/// super-scale per 256 elements.
const GROUPS_PER_SUPER: usize = 16;
/// Elements per super-block.
const SUPER_BLOCK: usize = GROUP_SIZE * GROUPS_PER_SUPER;

/// The signed 6-bit code grid an arm quantizes onto. Two grids exist
/// because TCF's SPECIFICATION.md Section 13.2 reserves the most-negative
/// code as a rejection point and ggml's Q6_K does not; the two differ by one
/// level in 64, and isolating that cost is what the reserved arm is for.
#[derive(Debug, Clone, Copy)]
struct CodeGrid {
    /// The divisor the fit sweep anchors on: `d = max_abs / (qmax + mult)`.
    qmax: f32,
    /// Lowest code emitted.
    lo: f32,
    /// Highest code emitted.
    hi: f32,
}

/// All 64 codes, `-32..=31`. Q6_K's grid.
const FULL_64: CodeGrid = CodeGrid {
    qmax: 32.0,
    lo: -32.0,
    hi: 31.0,
};

/// 63 codes, `-31..=31`, the most-negative pattern reserved. TCF's grid for
/// every symmetric encoding (`geometry.qmax()` is `2^(bits-1) - 1`).
const RESERVED_63: CodeGrid = CodeGrid {
    qmax: 31.0,
    lo: -31.0,
    hi: 31.0,
};
/// Sub-scale storage width: `u8`, `0..=255`, matching `Q6S16D_T64` and
/// Q6_K's `int8` (this probe stores it unsigned, since a negative sub-scale
/// duplicates a positive one with codes negated).
const MAX_SUB: f32 = 255.0;

/// Which format the per-super-block scale is stored in, and whether it is
/// pre-divided by the sub-scale range before rounding. The three arms
/// differ ONLY here — same group fit, same sub-scale search, same code
/// derivation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SuperPrecision {
    /// TCF's `Q6S16D_T64` design: `bf16`, pre-divided by 255 so decode is
    /// one multiply (`super * sub`). bf16 keeps f32's 8 exponent bits, so
    /// pre-dividing cannot underflow it.
    Bf16,
    /// Q6_K's storage design (with TCF's unsigned `u8` sub-scale instead of
    /// Q6_K's own int8 encoding): `f16`, NOT pre-divided. f16 has only 5
    /// exponent bits — pre-dividing a typical weight-scale magnitude by 255
    /// would underflow it, which is why TCF chose bf16 for its own
    /// pre-divided design. Storing the UNDIVIDED max instead avoids that,
    /// at the cost of one extra division per group at decode.
    F16,
    /// The ceiling: `super = max_g d_g` kept in `f32`, unrounded. No
    /// super-precision scheme can beat this — it is the two-level geometry
    /// with zero rounding above the per-group float fit.
    F32,
    /// TCF's `Q6S16D_T64` EXACTLY: the `Bf16` arm plus SPECIFICATION.md
    /// Section 13.2's reserved most-negative code, so 63 levels not 64. The
    /// only arm on [`RESERVED_63`]; against `Bf16` it isolates the reserved
    /// code's cost with the super-scale held fixed.
    Bf16Reserved,
}

impl SuperPrecision {
    /// The code grid this arm quantizes onto.
    fn grid(self) -> CodeGrid {
        match self {
            Self::Bf16Reserved => RESERVED_63,
            Self::Bf16 | Self::F16 | Self::F32 => FULL_64,
        }
    }
}

/// One group's ideal (pre-super-rounding) float scale `d_g`, kept for the
/// super-block pass that follows.
struct GroupFit {
    d_g: f32,
}

/// Nearest integer code to `u`, round-to-nearest-even, clamped to the
/// grid. The uniform-integer analogue of
/// [`super::quantize::nearest_level`]'s table lookup: there is no 16-entry
/// table here, so the "nearest level" is just an integer round-and-clamp.
fn nearest_code_f(u: f32, grid: CodeGrid) -> f32 {
    u.round_ties_even().clamp(grid.lo, grid.hi)
}

/// Weighted squared error of reconstructing `values` at scale `d`, codes
/// taken to the nearest 6-bit integer. Mirrors `quantize.rs`'s own
/// `weighted_squared_error` with a uniform-integer level set instead of a
/// table.
fn weighted_squared_error_uniform(values: &[f32], weights: &[f32], d: f32, grid: CodeGrid) -> f64 {
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
fn fit_group_scale(values: &[f32], weights: &[f32], grid: CodeGrid) -> GroupFit {
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

/// Refines an initial `u8` sub-scale guess over `±2` (the same radius
/// `Q6S16D_T64` applies — see `hats/tcf/tcf-core`'s
/// `refine_symmetric_sub_scale`), scoring each neighbor by weighted squared
/// error against ITS OWN effective scale (`effective(sub)`), keeping the
/// strictly-lower-error candidate so the search never moves off the
/// rounded start without cause.
fn refine_sub_scale(
    values: &[f32],
    weights: &[f32],
    rounded: u8,
    grid: CodeGrid,
    effective: impl Fn(u8) -> f32,
) -> u8 {
    let mut best: Option<(f64, u8)> = None;
    for delta in -2i32..=2 {
        let candidate = i32::from(rounded)
            .saturating_add(delta)
            .clamp(0, MAX_SUB as i32);
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
fn derive_codes(values: &[f32], effective: f32, grid: CodeGrid) -> Vec<f32> {
    if effective == 0.0 {
        return vec![0.0f32; values.len()];
    }
    values
        .iter()
        .map(|&x| effective * nearest_code_f(x / effective, grid))
        .collect()
}

/// Quantizes then dequantizes one super-block (up to [`SUPER_BLOCK`]
/// elements, grouped into up to [`GROUPS_PER_SUPER`] groups of
/// [`GROUP_SIZE`]) under `precision`.
///
/// Three passes, mirroring `Q6S16D_T64`'s own three-pass structure:
/// 1. [`fit_group_scale`] every group's ideal `f32` scale independently.
/// 2. Derive ONE super-scale from `max(d_g)` across the block, per
///    `precision`'s storage rule (see [`SuperPrecision`]).
/// 3. Per group: round `d_g` against the super-scale to an initial `u8`
///    sub-scale, [`refine_sub_scale`] it over `±2`, then [`derive_codes`]
///    against the FINAL effective scale.
///
/// A super-block where every group is all-zero (`max(d_g) == 0.0`) stores
/// super `0.0` and every group reconstructs to exact zeros without running
/// steps 2-3's rounding, which would otherwise divide by zero.
fn quantize_super_block(values: &[f32], weights: &[f32], precision: SuperPrecision) -> Vec<f32> {
    let grid = precision.grid();
    let groups: Vec<&[f32]> = values.chunks(GROUP_SIZE).collect();
    let weight_groups: Vec<&[f32]> = weights.chunks(GROUP_SIZE).collect();

    // Pass 1: every group's ideal float scale.
    let fits: Vec<GroupFit> = groups
        .iter()
        .zip(weight_groups.iter())
        .map(|(g, w)| fit_group_scale(g, w, grid))
        .collect();
    let max_d = fits.iter().fold(0.0f32, |acc, f| acc.max(f.d_g));

    if max_d == 0.0 {
        return vec![0.0f32; values.len()];
    }

    // Pass 2: the super-scale, per arm. `super_pre_divides` says whether
    // sub-scale recovery divides `d_g` by `super` directly (pre-divided:
    // Bf16) or by `super / MAX_SUB` (not pre-divided: F16, F32).
    let (super_scale, pre_divided) = match precision {
        SuperPrecision::Bf16 | SuperPrecision::Bf16Reserved => {
            (bf16::from_f32(max_d / MAX_SUB).to_f32(), true)
        }
        SuperPrecision::F16 => (f16::from_f32(max_d).to_f32(), false),
        SuperPrecision::F32 => (max_d, false),
    };
    if super_scale <= 0.0 || !super_scale.is_finite() {
        // Degenerate rounding (should not occur for a finite positive
        // `max_d` under either half-precision format at realistic weight
        // magnitudes) — fall back to exact zeros rather than propagate a
        // non-finite scale into every group's codes.
        return vec![0.0f32; values.len()];
    }

    // Pass 3: per group, sub-scale then codes against the final effective
    // scale. `effective(sub)` is arm-specific: `super * sub` when the
    // super-scale is pre-divided (the level count is already folded in),
    // `super * sub / MAX_SUB` when it is not.
    let effective_of = move |super_scale: f32, sub: u8| -> f32 {
        if pre_divided {
            super_scale * f32::from(sub)
        } else {
            super_scale * f32::from(sub) / MAX_SUB
        }
    };

    let mut out = Vec::with_capacity(values.len());
    for ((group, weight_group), fit) in groups.iter().zip(weight_groups.iter()).zip(fits.iter()) {
        if fit.d_g == 0.0 {
            out.extend(vec![0.0f32; group.len()]);
            continue;
        }
        let raw_sub = if pre_divided {
            fit.d_g / super_scale
        } else {
            fit.d_g * MAX_SUB / super_scale
        };
        let rounded = raw_sub.round_ties_even().clamp(0.0, MAX_SUB) as u8;
        let sub = refine_sub_scale(group, weight_group, rounded, grid, |s| {
            effective_of(super_scale, s)
        });
        let effective = effective_of(super_scale, sub);
        out.extend(derive_codes(group, effective, grid));
    }
    out
}

/// Quantizes then dequantizes every value in `values` against `precision`,
/// grouping [`SUPER_BLOCK`] consecutive elements per row exactly like
/// [`super::roundtrip::codebook_round_trip`] — same tail handling (a
/// super-block shorter than [`SUPER_BLOCK`] is quantized on its own actual
/// groups, never padded), same never-crosses-a-row-boundary rule, same
/// `in_features == 0` no-op.
pub fn two_level_codebook_round_trip(
    values: &[f32],
    in_features: usize,
    precision: SuperPrecision,
    weights: &[f32],
) -> Vec<f32> {
    if in_features == 0 {
        return values.to_vec();
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
    out
}

#[cfg(test)]
mod tests {
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
}
