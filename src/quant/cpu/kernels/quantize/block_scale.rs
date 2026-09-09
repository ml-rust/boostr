//! Whole-block scale search for the single-scale symmetric formats
//!
//! Q4_0 and Q8_0 store ONE f16 scale per 32-element block and nothing else. The
//! scale is still a free parameter. Moving it off the absmax fit changes which
//! level each element rounds to. A different level assignment can reconstruct
//! the block more closely, even with the largest magnitude off the top level.
//! Absmax is one point in that space, not its optimum.
//!
//! The sweep has the shape of
//! [`make_qx_quants`](super::search::make_qx_quants): candidate scales spaced
//! around the absmax choice, each refined by a least-squares refit for the
//! scale that best reconstructs its own level assignment. Three differences make
//! this a separate routine:
//!
//! - **Objective.** `make_qx_quants` weights by `x²`, llama.cpp's stand-in for
//!   an activation-magnitude prior over a K-quant sub-block. These formats carry
//!   no imatrix, and Q8_0 also holds activation-side data, so no evidence
//!   supports that prior here. The objective is unweighted squared
//!   reconstruction error, accumulated in f64.
//! - **The scored scale is the STORED one.** `make_qx_quants` returns an f32
//!   scale that Q6_K re-quantizes afterwards. Here each candidate rounds to
//!   binary16 FIRST, and the block scores against the exact value the reader
//!   loads. A scale that scores well in f32 and rounds badly is not a win.
//! - **Level range.** `make_qx_quants` clamps to `[-nmax, nmax - 1]` and emits
//!   biased unsigned levels. Q8_0's codes are `[-127, 127]` — symmetric, one
//!   short of the signed-byte range — which that routine cannot express.
//!
//! The absmax scale is candidate `is == 0`. A candidate with an unusable refit
//! falls back to its own unrefined scale. The search therefore never returns a
//! worse scale than absmax under this objective.
//!
//! # This diverges from llama.cpp on purpose
//!
//! llama.cpp's `quantize_row_q4_0` and `quantize_row_q8_0` are plain absmax fits
//! with no search. Every block whose search picks a different scale writes
//! different bytes than llama.cpp does. The files stay VALID Q4_0/Q8_0 — same
//! block size, field layout, code range and byte count — and every reader
//! including llama.cpp's decodes them. They are not bit-identical to llama.cpp's
//! output. This trades byte-for-byte reproduction of llama.cpp for lower
//! reconstruction error at identical file size.
//!
//! Q4_0 also changes its rounding rule. llama.cpp truncates `+ 8.5`, which is
//! round-half-up. [`block_code`] rounds ties to even, like every other writer
//! here. That moves a code only on an exact half. It is a second reason the
//! bytes cannot match.

use super::search::{GROUP_MAX_EPS, nearest_int, signed_absmax};
use half::f16;

/// Candidate scales swept either side of the absmax fit
///
/// Same width as `make_qx_quants`' sweep, so both searches explore
/// neighbourhoods of one size.
const SWEEP: i32 = 9;

/// Spacing between candidates, in units of one level
const STEP: f32 = 0.1;

/// Level range and starting scale of a single-scale symmetric block format
pub struct BlockScaleFit {
    /// Divisor applied to the anchor to get the starting scale
    pub nlevels: f32,
    /// Most negative code the format stores
    pub qmin: i32,
    /// Most positive code the format stores
    pub qmax: i32,
    /// Whether the anchor is the negated SIGNED absmax carrier
    ///
    /// Q4_0's code range is one step longer on the negative side. Its scale
    /// therefore takes the OPPOSITE sign to the largest-magnitude element, which
    /// puts that element on code -8 instead of wasting the extra step. Q8_0's
    /// range is symmetric, so its anchor is the magnitude alone and its scale is
    /// always positive.
    pub signed_anchor: bool,
}

/// Q4_0 — codes `[-8, 7]`, scale `max / -8`, stored biased by `+8`
pub const Q4_0_FIT: BlockScaleFit = BlockScaleFit {
    nlevels: 8.0,
    qmin: -8,
    qmax: 7,
    signed_anchor: true,
};

/// Q8_0 — codes `[-127, 127]`, scale `amax / 127`
pub const Q8_0_FIT: BlockScaleFit = BlockScaleFit {
    nlevels: 127.0,
    qmin: -127,
    qmax: 127,
    signed_anchor: false,
};

/// The code one element gets against the STORED scale
///
/// Every level in this module goes through here — the refit's assignment, the
/// scoring pass and the bytes the writer emits. None of them can silently
/// disagree about what a code is. A zero scale reconstructs zero whatever the
/// codes say, so it emits zeros rather than dividing by it.
#[inline]
pub fn block_code(v: f32, d: f32, fit: &BlockScaleFit) -> i32 {
    if d == 0.0 {
        return 0;
    }
    nearest_int(v / d).clamp(fit.qmin, fit.qmax)
}

/// The stored f16 scale that minimises squared reconstruction error, over the
/// swept candidates
///
/// Returns zero for an all-zero block, and when every candidate underflows or
/// overflows binary16. In both cases the codes are all zero and the block
/// reconstructs as zero, the only thing the format represents there.
pub fn fit_block_scale(x: &[f32], fit: &BlockScaleFit) -> f16 {
    let Some(anchor) = block_anchor(x, fit) else {
        return f16::ZERO;
    };

    let mut best: Option<(f64, u16)> = None;
    for is in -SWEEP..=SWEEP {
        let Some(d0) = candidate_scale(anchor, fit.nlevels, is) else {
            continue;
        };
        let d = refit(x, d0, fit);
        let err = squared_error(x, d.to_f32(), fit);
        // Ties resolve to the smaller bit pattern, so the winner never depends
        // on enumeration order.
        let improves = match best {
            None => true,
            Some((best_err, best_bits)) => {
                err < best_err || (err == best_err && d.to_bits() < best_bits)
            }
        };
        if improves {
            best = Some((err, d.to_bits()));
        }
    }

    best.map_or(f16::ZERO, |(_, bits)| f16::from_bits(bits))
}

/// The plain absmax scale — the choice [`fit_block_scale`] replaces
///
/// Tests measure the search against it through the identical encoder, which
/// isolates the sweep as the only difference. No shipped path calls it.
#[cfg(test)]
pub fn absmax_block_scale(x: &[f32], fit: &BlockScaleFit) -> f16 {
    match block_anchor(x, fit) {
        Some(anchor) => f16::from_f32(anchor / fit.nlevels),
        None => f16::ZERO,
    }
}

/// Numerator of the starting scale, or `None` for an all-zero block
fn block_anchor(x: &[f32], fit: &BlockScaleFit) -> Option<f32> {
    let (amax, signed_max) = signed_absmax(x);
    if amax < GROUP_MAX_EPS {
        return None;
    }
    Some(if fit.signed_anchor { -signed_max } else { amax })
}

/// Candidate `is`'s stored scale, or `None` when it leaves binary16's range
///
/// A candidate that underflows to zero or overflows to infinity cannot be
/// stored, so it is dropped rather than scored.
fn candidate_scale(anchor: f32, nlevels: f32, is: i32) -> Option<f16> {
    let d = f16::from_f32(anchor / (nlevels + STEP * is as f32));
    let v = d.to_f32();
    (v != 0.0 && v.is_finite()).then_some(d)
}

/// Least-squares refit of one candidate: the scale that best reconstructs the
/// level assignment `d0` produced
///
/// `d = Σ(x·l) / Σ(l²)`, accumulated in f64 because the sums run over the whole
/// block and cancel. A refit that underflows, overflows or flips sign is no
/// longer the same fit, so the candidate stands on `d0`. That keeps the plain
/// absmax scale reachable at `is == 0`.
fn refit(x: &[f32], d0: f16, fit: &BlockScaleFit) -> f16 {
    let d0f = d0.to_f32();
    let mut sum_xl = 0.0f64;
    let mut sum_ll = 0.0f64;
    for &v in x {
        let l = f64::from(block_code(v, d0f, fit));
        sum_xl += f64::from(v) * l;
        sum_ll += l * l;
    }
    if sum_ll <= 0.0 {
        return d0;
    }
    let refined = f16::from_f32((sum_xl / sum_ll) as f32);
    let rf = refined.to_f32();
    if rf.is_finite() && rf * d0f > 0.0 {
        refined
    } else {
        d0
    }
}

/// `Σ(x − d·q)²` with `q` the code the writer emits for `d`
fn squared_error(x: &[f32], d: f32, fit: &BlockScaleFit) -> f64 {
    let mut err = 0.0f64;
    for &v in x {
        let q = f64::from(block_code(v, d, fit));
        let diff = f64::from(v) - f64::from(d) * q;
        err += diff * diff;
    }
    err
}
