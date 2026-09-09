//! Whole-block scale-and-offset search for the affine block formats
//!
//! Q4_1 stores TWO binary16 fields per 32-element block — `d` at bytes 0..2 and
//! `m` at bytes 2..4 — and the reader reconstructs `d·q + m` with `q` an
//! unsigned code in `[0, 15]`. Both fields are free parameters. Moving them off
//! the plain min/max fit changes which level each element rounds to, and a
//! different level assignment can reconstruct the block more closely. Min/max is
//! one point in that space, not its optimum.
//!
//! # Why this is not [`super::block_scale`]
//!
//! [`super::block_scale`] models `d·q` alone: one free parameter, a symmetric
//! signed code range, and an anchor taken from the largest magnitude in the
//! block. Q4_1 differs on every one of those. Its offset is a second free
//! parameter, its codes are unsigned and one-sided, and its starting point comes
//! from the block's RANGE rather than its absmax. The refit is a two-by-two
//! normal-equation solve, not a single ratio, and the scoring pass reconstructs
//! through two stored fields. Nothing of `block_scale`'s single-parameter shape
//! survives that, so this is a sibling routine rather than a generalisation of
//! it.
//!
//! # Why this is not [`make_qkx2_quants`](super::search::make_qkx2_quants)
//!
//! That routine's offset is FORCED non-positive, because Q4_K and Q5_K store an
//! unsigned `dmin` that the reader SUBTRACTS. Q4_1's `m` is signed and added, so
//! clamping it away discards half the format's expressive range — an
//! all-positive block would be pinned to an offset of zero when its own minimum
//! is the fit that reconstructs it. `make_qkx2_quants` is also weighted, and
//! scores an f32 scale that the K-quant writers requantize afterwards, where
//! Q4_1's fields go to the file as-is.
//!
//! The sweep's SHAPE is taken from those two routines: candidates spaced a
//! fraction of a level either side of the direct fit, each refined by a
//! least-squares solve for the parameters that best reconstruct its own level
//! assignment. Three properties are specific to this module:
//!
//! - **Objective.** Unweighted squared reconstruction error, accumulated in f64.
//!   Q4_1 carries no imatrix, so no activation-magnitude prior is available to
//!   weight by.
//! - **The scored parameters are the STORED ones.** Both `d` and `m` round to
//!   binary16 BEFORE the block is scored, so a pair that fits well in f32 and
//!   rounds badly cannot win. Codes likewise divide by the rounded `d` and
//!   subtract the rounded `m`, never the wider floats they came from.
//! - **The direct fit is scored, not merely reachable.** Each candidate offers
//!   its unrefined pair as well as its refit, and candidate `is == 0`'s
//!   unrefined pair IS the plain min/max fit. The search therefore cannot score
//!   worse than the fit it replaces.
//!
//! # This diverges from llama.cpp on purpose
//!
//! llama.cpp's `quantize_row_q4_1` is a plain min/max fit with no search. Every
//! block whose search picks a different pair writes different bytes than
//! llama.cpp does. The files stay VALID Q4_1 — same block size, field layout,
//! code range and byte count — and every reader including llama.cpp's decodes
//! them. This trades byte-for-byte reproduction of llama.cpp for lower
//! reconstruction error at identical file size, the same trade
//! [`super::block_scale`] states for Q4_0 and Q8_0.
//!
//! Q4_1 also changes its rounding rule. llama.cpp truncates `+ 0.5`, which is
//! round-half-up. [`affine_code`] rounds ties to even, like every other writer
//! here. That moves a code only on an exact half. It is a second reason the
//! bytes cannot match.

use super::search::nearest_int;
use half::f16;

/// Candidate ranges swept either side of the direct fit
///
/// Same width as the symmetric sweep, so both searches explore neighbourhoods of
/// one size.
const SWEEP: i32 = 9;

/// Spacing between candidates, in units of one level
const STEP: f32 = 0.1;

/// Level range of an affine block format: `x ≈ d·q + m`, `q ∈ [qmin, qmax]`
pub struct BlockAffineFit {
    /// Number of level STEPS the code range spans, the divisor of the block range
    pub nlevels: f32,
    /// Lowest code the format stores
    pub qmin: i32,
    /// Highest code the format stores
    pub qmax: i32,
}

/// Q4_1 — unsigned codes `[0, 15]`, so fifteen steps across the block range
pub const Q4_1_FIT: BlockAffineFit = BlockAffineFit {
    nlevels: 15.0,
    qmin: 0,
    qmax: 15,
};

/// The code one element gets against the STORED `d` and `m`
///
/// Every level in this module goes through here — the refit's assignment, the
/// scoring pass and the bytes the writer emits. None of them can silently
/// disagree about what a code is. A zero `d` reconstructs `m` whatever the codes
/// say, so it emits the bottom code rather than dividing by it.
#[inline]
pub fn affine_code(v: f32, d: f32, m: f32, fit: &BlockAffineFit) -> i32 {
    if d == 0.0 {
        return fit.qmin;
    }
    nearest_int((v - m) / d).clamp(fit.qmin, fit.qmax)
}

/// The stored binary16 `(d, m)` pair that minimises squared reconstruction
/// error, over the swept candidates
///
/// Returns the plain min/max pair when no candidate is storable, which is what
/// the writer emitted before the search existed. An empty block has nothing to
/// fit and returns zeros.
pub fn fit_block_affine(x: &[f32], fit: &BlockAffineFit) -> (f16, f16) {
    let Some((min, max)) = block_range(x) else {
        return (f16::ZERO, f16::ZERO);
    };
    let direct = stored_pair(min, max, fit.nlevels);

    let mut best: Option<(f64, u16, u16)> = None;
    for is in -SWEEP..=SWEEP {
        let levels = fit.nlevels + STEP * is as f32;
        if levels <= 0.0 {
            continue;
        }
        let (d0, m0) = stored_pair(min, max, levels);
        // The unrefined pair is scored alongside its refit, so candidate
        // `is == 0` puts the plain min/max fit itself in the pool.
        consider(x, fit, d0, m0, &mut best);
        if let Some((d, m)) = refit(x, d0, m0, fit) {
            consider(x, fit, d, m, &mut best);
        }
    }

    best.map_or(direct, |(_, d, m)| (f16::from_bits(d), f16::from_bits(m)))
}

/// The plain min/max fit — the choice [`fit_block_affine`] replaces
///
/// Tests measure the search against it through the identical encoder, which
/// isolates the sweep as the only difference. No shipped path calls it.
#[cfg(test)]
pub fn minmax_block_affine(x: &[f32], fit: &BlockAffineFit) -> (f16, f16) {
    match block_range(x) {
        Some((min, max)) => stored_pair(min, max, fit.nlevels),
        None => (f16::ZERO, f16::ZERO),
    }
}

/// Smallest and largest value in the block, or `None` when there is no block
fn block_range(x: &[f32]) -> Option<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        min = min.min(v);
        max = max.max(v);
    }
    (min <= max).then_some((min, max))
}

/// The stored pair that spreads `[min, max]` over `levels` steps
///
/// `m` is the block minimum, so code zero reconstructs it and the offset is
/// ADDED. A constant block gives `d == 0`, whose codes are all zero and whose
/// reconstruction is `m` — the only value the format represents there.
fn stored_pair(min: f32, max: f32, levels: f32) -> (f16, f16) {
    (f16::from_f32((max - min) / levels), f16::from_f32(min))
}

/// Scores one stored pair and keeps it when it beats the incumbent
///
/// Ties resolve to the smaller `(d, m)` bit patterns, so the winner never
/// depends on enumeration order. A pair that does not survive binary16 is
/// dropped rather than scored: no reader can use it.
fn consider(x: &[f32], fit: &BlockAffineFit, d: f16, m: f16, best: &mut Option<(f64, u16, u16)>) {
    let (df, mf) = (d.to_f32(), m.to_f32());
    if !df.is_finite() || !mf.is_finite() {
        return;
    }
    let err = squared_error(x, df, mf, fit);
    let (db, mb) = (d.to_bits(), m.to_bits());
    let improves = match *best {
        None => true,
        Some((best_err, best_d, best_m)) => {
            err < best_err || (err == best_err && (db, mb) < (best_d, best_m))
        }
    };
    if improves {
        *best = Some((err, db, mb));
    }
}

/// Least-squares refit of one candidate: the `(d, m)` pair that best
/// reconstructs the level assignment `(d0, m0)` produced
///
/// The unweighted normal equations for `x ≈ d·l + m` over `n` elements are
///
/// ```text
/// det = n·Σl² − (Σl)²
/// d   = (n·Σ(x·l) − Σx·Σl) / det
/// m   = (Σl²·Σx − Σl·Σ(x·l)) / det
/// ```
///
/// accumulated in f64 because the sums run over the whole block and cancel. A
/// zero determinant means every element landed on one level, which fixes no
/// scale. A refit that leaves binary16's range or flips `d`'s sign is no longer
/// the same fit. Either way the candidate stands on its unrefined pair, which
/// [`fit_block_affine`] has already scored.
fn refit(x: &[f32], d0: f16, m0: f16, fit: &BlockAffineFit) -> Option<(f16, f16)> {
    let (d0f, m0f) = (d0.to_f32(), m0.to_f32());
    if !d0f.is_finite() || !m0f.is_finite() {
        return None;
    }

    let n = x.len() as f64;
    let (mut sum_l, mut sum_ll, mut sum_x, mut sum_xl) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &v in x {
        let l = f64::from(affine_code(v, d0f, m0f, fit));
        let xv = f64::from(v);
        sum_l += l;
        sum_ll += l * l;
        sum_x += xv;
        sum_xl += xv * l;
    }

    let det = n * sum_ll - sum_l * sum_l;
    if !det.is_finite() || det <= 0.0 {
        return None;
    }
    let d = (n * sum_xl - sum_x * sum_l) / det;
    let m = (sum_ll * sum_x - sum_l * sum_xl) / det;

    let refined_d = f16::from_f32(d as f32);
    let refined_m = f16::from_f32(m as f32);
    (refined_d.to_f32() * d0f > 0.0).then_some((refined_d, refined_m))
}

/// `Σ(x − (d·q + m))²` with `q` the code the writer emits for `(d, m)`
fn squared_error(x: &[f32], d: f32, m: f32, fit: &BlockAffineFit) -> f64 {
    let mut err = 0.0f64;
    for &v in x {
        let q = f64::from(affine_code(v, d, m, fit));
        let diff = f64::from(v) - (f64::from(d) * q + f64::from(m));
        err += diff * diff;
    }
    err
}
