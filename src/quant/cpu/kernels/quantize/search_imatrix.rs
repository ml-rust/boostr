//! Search machinery the importance-weighted (imatrix) K-quant writers add
//!
//! `ggml-quants.c` carries two writers per K-quant: `quantize_row_*_K_ref`,
//! which the no-imatrix path takes, and `quantize_row_*_K_impl`, which
//! `quantize_*_K` takes the moment an importance matrix is supplied. Nearly
//! every GGUF quant the ecosystem ships comes out of the second one, so boostr
//! ships both. [`super::search`] holds what the two paths share; this module
//! holds what only the second one needs.
//!
//! # `make_qkx3_quants` is `make_qkx2_quants`
//!
//! The `_impl` writers for Q2_K, Q4_K and Q5_K call `make_qkx3_quants` where
//! the `_ref` writers call `make_qkx2_quants`. The two C routines differ in
//! exactly two places, and neither is a difference in behaviour for these
//! callers:
//!
//! - `make_qkx3_quants` accepts a NULL `weights` and falls back to `w = x²`.
//!   Every K-quant `_impl` call site passes a non-NULL weights array, so the
//!   branch is unreachable from here.
//! - `make_qkx3_quants` exits early on `max <= min`, `make_qkx2_quants` on
//!   `max == min`. Both run after `min` is clamped to `≤ 0`, so `max < min`
//!   requires `max < min ≤ 0`, i.e. a maximum below the minimum. The two
//!   conditions are the same condition.
//!
//! So [`super::search::make_qkx2_quants`] IS `make_qkx3_quants`, and the
//! imatrix formats reach it with their own [`super::search::KSearch`]
//! constants: `rmin = -0.9`, `rdelta = 0.05`, `nstep = 36` and
//! `use_mad = false` for all three, against the `-1.0/0.1/20`, `-0.5/0.1/15`
//! and `-0.5/0.1/15 + use_mad` of the `_ref` path. A second copy of the sweep
//! would be a second place for those constants to rot.
//!
//! # Degenerate importance entries
//!
//! A zero importance entry is legal and llama.cpp handles it without a special
//! case, so neither does this module. It zeroes that element's weight, which
//! costs it its vote in every sum below. Taken to the limit — a sub-block whose
//! importance is all zeros — every weighted sum is zero, and each of the three
//! routines already refuses to divide by it: [`make_qp_quants`] returns `0.0`
//! unless `suml2 > 0`, `make_qkx2_quants` skips a candidate unless its
//! determinant is positive, and [`make_qx_quants_weighted`] returns `0.0`
//! unless `suml2` is non-zero. The result is a zero scale, which the writers
//! already treat as an all-zero sub-block. No NaN, no divide by zero.
//!
//! Non-finite and negative importance entries have no such reading: a NaN
//! weight would poison every sum it touches and reach the file as a NaN scale
//! no reader can use, and an importance is a mean square activation, which is
//! never negative. Both are rejected as an error at the `QuantizeOps` boundary
//! rather than being silently repaired here, and so is an importance vector of
//! the wrong length — a silently ignored importance vector produces a file
//! indistinguishable from an unweighted one.

use super::search::{GROUP_MAX_EPS, nearest_int, qx_quants};

/// Symmetric scale search with caller-supplied weights — llama.cpp
/// `make_qx_quants` with a non-NULL `qw`
///
/// Identical to [`super::search::make_qx_quants`] in every respect but the
/// weight: that one uses `w = x²`, this one uses `w = qw[i]`. Q3_K's imatrix
/// path passes the shared `qw · sqrt(sigma2 + x²)` weight, and also reuses this
/// routine one level up to fit the super-block scale against the 16 sub-block
/// weights. Q6_K's imatrix path passes the RAW importance — see
/// [`super::q6k::quantize_q6k_imatrix`], which documents why.
pub fn make_qx_quants_weighted(x: &[f32], nmax: i32, levels: &mut [u8], qw: &[f32]) -> f32 {
    qx_quants(x, nmax, levels, Some(qw))
}

/// One-sided weighted scale search — llama.cpp `make_qp_quants`
///
/// Fits `x ≈ scale · l` with `l ∈ [0, nmax]` over NON-NEGATIVE `x`, and returns
/// the scale. Q2_K, Q4_K and Q5_K use it in their imatrix paths to quantize the
/// per-sub-block scales and mins against one super-block factor, weighted by
/// `sw` — the sum of each sub-block's own element weights, so a sub-block that
/// matters more to the row gets its scale represented more faithfully. The
/// `_ref` path instead divides by the maximum and rounds, which weights every
/// sub-block the same.
///
/// Three stages, all from the reference:
///
/// 1. sweep 8 candidate scales either side of the `nmax/max` fit and keep the
///    one with the lowest weighted squared error,
/// 2. take the least-squares optimum `Σw·x·l / Σw·l²` for that assignment,
/// 3. five passes of coordinate descent over the levels, moving element `i`
///    only when the objective `(Σw·x·l)² / Σw·l²` rises, exactly as
///    [`super::search::make_q3_quants`] does.
///
/// Levels are written to `levels` UNBIASED — this grid is one-sided, so there
/// is no bias to subtract — and truncated to `u8` as the C does. `x` is a
/// scale or a min, both of which the search that produced them keeps
/// non-negative, so the truncation is not reached on any input the writers
/// hand it; a negative value would wrap the same way in C, and the callers
/// mask or clamp the stored field afterwards.
pub fn make_qp_quants(x: &[f32], nmax: i32, levels: &mut [u8], sw: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 || levels.len() < n || sw.len() < n {
        return 0.0;
    }

    let max = x.iter().fold(0.0f32, |acc, &v| acc.max(v));
    if max < GROUP_MAX_EPS {
        levels[..n].fill(0);
        return 0.0;
    }

    // Candidate sweep. The error is scored against the TRUNCATED level, which
    // is what the reader will multiply, so the sweep cannot prefer a scale that
    // only looks good before truncation.
    let mut iscale = nmax as f32 / max;
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        levels[i] = nearest_int(iscale * x[i]) as u8;
        let diff = x[i] - scale * levels[i] as f32;
        best_mse += sw[i] * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * l as f32;
            mse += sw[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    // Least-squares optimum for the winning assignment.
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).min(nmax);
        levels[i] = l as u8;
        sumlx += sw[i] * x[i] * l as f32;
        suml2 += sw[i] * (l * l) as f32;
    }

    // Coordinate descent over the levels.
    for _ in 0..5 {
        let mut changed = 0usize;
        for i in 0..n {
            let w = sw[i];
            let cur = levels[i] as i32;
            let slx = sumlx - w * x[i] * cur as f32;
            let sl2 = suml2 - w * (cur * cur) as f32;
            if slx <= 0.0 || sl2 <= 0.0 {
                continue;
            }
            let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
            if new_l == cur {
                continue;
            }
            let slx = slx + w * x[i] * new_l as f32;
            let sl2 = sl2 + w * (new_l * new_l) as f32;
            // `slx²·suml2 > sumlx²·sl2` compares the two objectives without
            // dividing, the same trick the other searches use.
            if slx * slx * suml2 > sumlx * sumlx * sl2 {
                levels[i] = new_l as u8;
                sumlx = slx;
                suml2 = sl2;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }

    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// The importance entries covering super-block `b`
///
/// The importance vector is one entry per COLUMN of the weight matrix, so its
/// length is the row length and every row of the tensor indexes the same
/// vector — `quant_weights + QK_K*i` in `ggml-quants.c`, where `i` counts
/// super-blocks within one row and the pointer never advances between rows.
/// A row length is a multiple of the 256-element super-block, so a super-block
/// never straddles the wrap.
pub fn block_importance(imatrix: &[f32], b: usize, super_block: usize) -> &[f32] {
    let offset = (b * super_block) % imatrix.len();
    &imatrix[offset..][..super_block]
}
