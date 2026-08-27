//! Per-sub-block scale search — the reason K-quant output beats plain absmax
//!
//! A block format stores one scale per sub-block and a small integer per
//! element. The naive choice is absmax: `scale = max|x| / nmax`, which forces
//! the single largest magnitude in the sub-block to land exactly on the top
//! level and lets every other element absorb the rounding. That is the choice
//! compressr shipped, and it measured 7.85% relative RMS on a Q4_K conversion
//! where llama.cpp's own output measured 7.15% — same format, same block
//! geometry, same theoretical floor, ~9% of the accuracy given away.
//!
//! llama.cpp instead SEARCHES: it sweeps a small family of candidate scales
//! around the absmax choice, and for each candidate solves the weighted
//! least-squares problem for the scale that actually minimises reconstruction
//! error given that assignment of elements to levels. The two routines here are
//! ports of `make_qx_quants` (symmetric, used by Q6_K) and `make_qkx2_quants`
//! (scale + min, used by Q4_K and Q5_K) from `ggml-quants.c`.
//!
//! Both are weighted: the weight decides which elements the scale is allowed to
//! disappoint. `make_qx_quants` uses `w = x²`, so large weights dominate.
//! `make_qkx2_quants` is called with `w = sqrt(Σx²/n) + |x|`, which keeps small
//! elements from being written off entirely.

/// Below this magnitude a sub-block is treated as all-zero
///
/// llama.cpp's `GROUP_MAX_EPS`. Guards the `1/max` in both routines.
pub const GROUP_MAX_EPS: f32 = 1e-15;

/// Largest sub-block the asymmetric search handles (Q4_K/Q5_K use 32)
pub const MAX_SUB_BLOCK: usize = 32;

/// Round to nearest, ties to even — llama.cpp's `nearest_int`
///
/// llama.cpp implements this with the `+ 12582912.0f` magic-constant trick,
/// which is exactly IEEE round-half-to-even. Using `round()` here (ties away
/// from zero) would disagree with the reference on every exact `.5`.
#[inline]
pub fn nearest_int(v: f32) -> i32 {
    v.round_ties_even() as i32
}

/// Symmetric scale search over one sub-block — llama.cpp `make_qx_quants`
///
/// Returns the chosen scale and writes BIASED levels (`l + nmax`, so unsigned)
/// into `levels`. Levels are clamped to `[-nmax, nmax - 1]`: the negative side
/// gets one more step than the positive side, matching the two's-complement
/// range the readers subtract the bias back out of.
///
/// The search: start from the absmax scale, take the least-squares optimum for
/// that level assignment (`scale = Σw·x·l / Σw·l²`), then sweep 18 nearby
/// scales and keep whichever maximises `(Σw·x·l)² / Σw·l²` — the equivalent
/// maximisation, avoiding a division per candidate.
///
/// `weight = x²` (llama.cpp `rmse_type == 1`, which is what Q6_K passes).
pub fn make_qx_quants(x: &[f32], nmax: i32, levels: &mut [u8]) -> f32 {
    let (amax, max) = signed_absmax(x);
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }

    // Absmax starting point, then its least-squares optimum.
    let mut iscale = -(nmax as f32) / max;
    let (sumlx, suml2) = accumulate_symmetric(x, nmax, iscale, Some(&mut *levels));
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -((nmax as f32) + 0.1 * is as f32) / max;
        let (sx, s2) = accumulate_symmetric(x, nmax, iscale, None);
        // `sx² > best·s2` is `(sx/s2)·sx > best` without the division.
        if s2 > 0.0 && sx * sx > best * s2 {
            accumulate_symmetric(x, nmax, iscale, Some(&mut *levels));
            scale = sx / s2;
            best = scale * sx;
        }
    }

    scale
}

/// Absmax baseline — the scale choice the search replaces
///
/// Kept so tests can measure the search against it on identical input. This is
/// `make_qx_quants`' starting point with neither the least-squares correction
/// nor the sweep.
#[cfg(test)]
pub fn make_qx_absmax(x: &[f32], nmax: i32, levels: &mut [u8]) -> f32 {
    let (amax, max) = signed_absmax(x);
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    accumulate_symmetric(x, nmax, iscale, Some(levels));
    1.0 / iscale
}

/// Largest magnitude in `x`, and the SIGNED value carrying it
fn signed_absmax(x: &[f32]) -> (f32, f32) {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    (amax, max)
}

/// Accumulate `(Σw·x·l, Σw·l²)` for one candidate scale, optionally recording levels
fn accumulate_symmetric(
    x: &[f32],
    nmax: i32,
    iscale: f32,
    mut levels: Option<&mut [u8]>,
) -> (f32, f32) {
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
        if let Some(out) = levels.as_mut() {
            out[i] = (l + nmax) as u8;
        }
        let w = v * v;
        sumlx += w * v * l as f32;
        suml2 += w * (l * l) as f32;
    }
    (sumlx, suml2)
}

/// Asymmetric scale + min search — llama.cpp `make_qkx2_quants`
///
/// Models the sub-block as `x ≈ scale·l + min` with `l ∈ [0, nmax]` and
/// `min ≤ 0`, and returns `(scale, -min)` — the SECOND element is the negated
/// minimum, i.e. what Q4_K/Q5_K store as a non-negative `dmin` scale factor and
/// the readers SUBTRACT (`out = dl·q - ml`).
///
/// For each candidate `iscale` the weighted least-squares system for scale and
/// min jointly is:
///
/// ```text
/// D          = Σw·Σ(w·l²) − (Σw·l)²
/// this_scale = (Σw·Σ(w·l·x) − Σ(w·x)·Σ(w·l)) / D
/// this_min   = (Σ(w·l²)·Σ(w·x) − Σ(w·l)·Σ(w·l·x)) / D
/// ```
///
/// A positive `this_min` is rejected (clamped to 0, scale refitted alone)
/// because the stored `dmin` is unsigned. Candidates are compared by weighted
/// SQUARED error; llama.cpp's `use_mad` variant is not used by these formats.
///
/// `laux` is caller-provided scratch of at least `x.len()` bytes so the sweep
/// allocates nothing per sub-block.
// Eight search parameters plus the input, none derivable from another: `rmin`,
// `rdelta` and `nstep` define the sweep, `nmax` the level range, and `weights`
// the error metric. llama.cpp passes different values per format (Q4_K sweeps
// 20 steps from -1.0, Q5_K sweeps 15 from -0.5), so they cannot be constants.
#[allow(clippy::too_many_arguments)]
pub fn make_qkx2_quants(
    x: &[f32],
    nmax: i32,
    weights: &[f32],
    levels: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    if n == 0 || weights.len() < n || levels.len() < n || laux.len() < n {
        return (0.0, 0.0);
    }

    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    // The stored min is unsigned and subtracted, so it can only pull values
    // DOWN — an all-positive sub-block gets min = 0, not min = x_min.
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        levels[..n].fill(0);
        return (0.0, -min);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_err = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        levels[i] = l as u8;
        let diff = scale * l as f32 + min - x[i];
        best_err += weights[i] * diff * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l * l) as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det <= 0.0 {
            continue;
        }
        let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
        let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
        if this_min > 0.0 {
            this_min = 0.0;
            this_scale = sum_xl / sum_l2;
        }
        let mut err = 0.0f32;
        for i in 0..n {
            let diff = this_scale * laux[i] as f32 + this_min - x[i];
            err += weights[i] * diff * diff;
        }
        if err < best_err {
            levels[..n].copy_from_slice(&laux[..n]);
            best_err = err;
            scale = this_scale;
            min = this_min;
        }
    }

    (scale, -min)
}
