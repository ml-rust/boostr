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
//! error given that assignment of elements to levels. The three routines here
//! are ports of `make_qx_quants` (symmetric, used by Q6_K), `make_q3_quants`
//! (symmetric, coordinate descent, used by Q3_K) and `make_qkx2_quants`
//! (scale + min, used by Q2_K, Q4_K and Q5_K) from `ggml-quants.c`. The
//! importance-weighted writers reuse all three and add one more routine of
//! their own — see [`super::search_imatrix`].
//!
//! All three are weighted: the weight decides which elements the scale is
//! allowed to disappoint. `make_qx_quants` and `make_q3_quants` use `w = x²`, so
//! large weights dominate. `make_qkx2_quants` is called with
//! `w = sqrt(Σx²/n) + |x|` by Q4_K and Q5_K, and with `w = |x|` by Q2_K; either
//! way small elements keep a vote instead of being written off entirely.
//!
//! The single-scale formats (Q4_0, Q8_0) sweep their one block scale the same
//! way, but against an unweighted objective and scoring the binary16 value the
//! reader actually loads. That search is [`super::block_scale`], not these two
//! — see its module docs for why it cannot reuse `make_qx_quants`.

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
    qx_quants(x, nmax, levels, None)
}

/// [`make_qx_quants`] with the weight source left open — `qw = None` is
/// llama.cpp's `rmse_type == 1` (`w = x²`), `qw = Some(..)` the same C routine's
/// `qw` argument, passed by [`super::search_imatrix::make_qx_quants_weighted`].
pub(super) fn qx_quants(x: &[f32], nmax: i32, levels: &mut [u8], qw: Option<&[f32]>) -> f32 {
    let (amax, max) = signed_absmax(x);
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }

    // Absmax starting point, then its least-squares optimum.
    let mut iscale = -(nmax as f32) / max;
    let (sumlx, suml2) = accumulate_symmetric(x, nmax, iscale, qw, Some(&mut *levels));
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -((nmax as f32) + 0.1 * is as f32) / max;
        let (sx, s2) = accumulate_symmetric(x, nmax, iscale, qw, None);
        // `sx² > best·s2` is `(sx/s2)·sx > best` without the division.
        if s2 > 0.0 && sx * sx > best * s2 {
            accumulate_symmetric(x, nmax, iscale, qw, Some(&mut *levels));
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
    accumulate_symmetric(x, nmax, iscale, None, Some(levels));
    1.0 / iscale
}

/// Largest magnitude in `x`, and the SIGNED value carrying it
pub(super) fn signed_absmax(x: &[f32]) -> (f32, f32) {
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

/// Accumulate `(Σw·x·l, Σw·l²)` for one candidate scale, optionally recording
/// levels. `qw = None` means `w = x²`, else the caller supplies the weight.
fn accumulate_symmetric(
    x: &[f32],
    nmax: i32,
    iscale: f32,
    qw: Option<&[f32]>,
    mut levels: Option<&mut [u8]>,
) -> (f32, f32) {
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
        if let Some(out) = levels.as_mut() {
            out[i] = (l + nmax) as u8;
        }
        let w = match qw {
            Some(q) => q[i],
            None => v * v,
        };
        sumlx += w * v * l as f32;
        suml2 += w * (l * l) as f32;
    }
    (sumlx, suml2)
}

/// Per-element error of the asymmetric search — squared, or absolute if `use_mad`
#[inline]
fn penalty(diff: f32, use_mad: bool) -> f32 {
    if use_mad { diff.abs() } else { diff * diff }
}

/// Symmetric coordinate-descent search over one sub-block — llama.cpp
/// `make_q3_quants` with `do_rmse = true`
///
/// Returns the chosen scale and writes BIASED levels (`l + nmax`) into
/// `levels`, exactly as [`make_qx_quants`] does. Q3_K is the only caller.
///
/// Where `make_qx_quants` sweeps the SCALE and re-derives every level for each
/// candidate, this sweeps the LEVELS one element at a time and lets the scale
/// fall out of them. For element `i` it removes `i`'s contribution from the
/// running `(Σw·x·l, Σw·l²)`, picks the level that best explains `x[i]` under
/// what is left, and keeps the move only when it raises the objective
/// `(Σw·x·l)² / Σw·l²`. Five passes, stopping early once a pass changes nothing.
///
/// `weight = x²`, matching `make_qx_quants`' `rmse_type == 1`.
///
/// The levels are working state, not the output: `quantize_q3k` re-derives
/// every level against the 6-bit scale the reader reconstructs. Only the
/// returned scale survives, except on a sub-block whose stored scale rounds to
/// zero — there the second pass is skipped and these levels are what ship.
pub fn make_q3_quants(x: &[f32], nmax: i32, levels: &mut [u8]) -> f32 {
    let n = x.len();
    if n == 0 || levels.len() < n {
        return 0.0;
    }
    let (amax, max) = signed_absmax(x);
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }

    // Absmax starting point. Levels are held BIASED in `levels` throughout, so
    // the routine needs no signed scratch buffer of its own.
    let iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = accumulate_symmetric(x, nmax, iscale, None, Some(&mut *levels));

    for _ in 0..5 {
        let mut changed = 0usize;
        for (i, &v) in x.iter().enumerate() {
            let l = levels[i] as i32 - nmax;
            let w = v * v;
            // Objective without element `i`.
            let slx = sumlx - w * v * l as f32;
            if slx <= 0.0 {
                continue;
            }
            let sl2 = suml2 - w * (l * l) as f32;
            let new_l = nearest_int(v * sl2 / slx).clamp(-nmax, nmax - 1);
            if new_l == l {
                continue;
            }
            let slx = slx + w * v * new_l as f32;
            let sl2 = sl2 + w * (new_l * new_l) as f32;
            // `slx²·suml2 > sumlx²·sl2` compares the two objectives without
            // dividing, the same trick `make_qx_quants` uses across its sweep.
            if slx * slx * suml2 > sumlx * sumlx * sl2 {
                levels[i] = (new_l + nmax) as u8;
                sumlx = slx;
                suml2 = sl2;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }

    // The element carrying `max` always lands on a non-zero level, so `suml2`
    // cannot be zero here. The guard keeps a NaN scale out of the file anyway.
    if suml2 != 0.0 { sumlx / suml2 } else { 0.0 }
}

/// Sweep parameters for the asymmetric scale+min search, per format
///
/// Taken from the `make_qkx2_quants` call sites in `ggml-quants.c`. The formats
/// do NOT share them: Q5_K's finer level grid needs a narrower sweep, and Q2_K
/// scores candidates by absolute rather than squared error.
pub struct KSearch {
    /// Top quantization level (`3` for 2-bit, `15` for 4-bit, `31` for 5-bit)
    pub nmax: i32,
    /// Lowest sweep offset applied to `nmax`
    pub rmin: f32,
    /// Sweep step
    pub rdelta: f32,
    /// Number of sweep steps; `0` disables the search (plain min/max fit)
    pub nstep: i32,
    /// Score candidates by weighted ABSOLUTE error instead of squared error
    ///
    /// llama.cpp's `use_mad`. Only Q2_K sets it. With two bits per element the
    /// squared metric lets one badly-placed outlier buy the scale, and the
    /// other fifteen elements pay for it.
    pub use_mad: bool,
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
/// SQUARED error, or by weighted ABSOLUTE error when `use_mad` is set —
/// llama.cpp passes `use_mad = true` for Q2_K only.
///
/// `laux` is caller-provided scratch of at least `x.len()` bytes so the sweep
/// allocates nothing per sub-block.
// Nine search parameters plus the input, none derivable from another: `rmin`,
// `rdelta` and `nstep` define the sweep, `nmax` the level range, `weights` and
// `use_mad` the error metric. llama.cpp passes different values per format
// (Q4_K sweeps 20 steps from -1.0, Q5_K and Q2_K sweep 15 from -0.5, and only
// Q2_K scores by absolute error), so they cannot be constants.
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
    use_mad: bool,
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
        best_err += weights[i] * penalty(diff, use_mad);
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
            err += weights[i] * penalty(diff, use_mad);
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

/// Mean square of a super-block — the `sumx2/QK_K` of every `_impl` writer. The
/// importance paths disagree on the factor applied to it — Q2_K takes
/// `sigma2 = Σx²/n`, Q3_K/Q4_K/Q5_K `2·Σx²/n`, Q6_K none — so it stays with the
/// caller.
pub fn block_sigma2(x: &[f32]) -> f32 {
    let sum_x2: f32 = x.iter().map(|v| v * v).sum();
    sum_x2 / x.len() as f32
}

/// Per-element weight of an importance-weighted sub-block, and its sum —
/// `w[l] = qw[l] · sqrt(sigma2 + x[l]²)`
///
/// The one expression Q2_K, Q3_K, Q4_K and Q5_K share in their `_impl` paths in
/// `ggml-quants.c`. The importance `qw` says how much the row's activations
/// care about each COLUMN; the `sqrt(sigma2 + x²)` term is the data's own say,
/// so a high-importance column with a tiny weight cannot dominate the sub-block
/// outright. The returned sum is llama.cpp's `sw[j]`, the weight the sub-block
/// carries when its own scale is quantized against the super-block factor. A
/// zero `qw` entry is legal and costs that element its vote, never a NaN —
/// [`super::search_imatrix`] documents why, and what is rejected instead.
pub fn importance_weights(x: &[f32], qw: &[f32], sigma2: f32, out: &mut [f32]) -> f32 {
    let mut sumw = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let w = qw[i] * (sigma2 + v * v).sqrt();
        out[i] = w;
        sumw += w;
    }
    sumw
}
