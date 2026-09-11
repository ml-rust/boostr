//! IQ4_NL quantization writer — llama.cpp `quantize_row_iq4_nl_impl`
//!
//! 32 elements, 18 bytes: f16 `d` then 16 nibble-pair bytes. Unlike Q4_0/Q8_0's
//! uniform integer levels, each nibble indexes [`KVALUES_IQ4NL`], a fixed
//! sorted non-linear codebook — nearest-entry lookup is
//! [`super::codebook::best_index_int8`], shared with the IQ4_XS writer.
//!
//! `quantize_row_iq4_nl_impl` in `ggml-quants.c` is generic over a
//! `super_block_size / block_size` ratio that [`super::iq4_xs`] uses (8
//! sub-blocks of 32 under one super-block scale). For IQ4_NL the two sizes are
//! equal — one 32-element block, one scale — so that branch never runs and is
//! not ported here.
//!
//! # The scale search — same shape as `d*sumqx` maximisation elsewhere
//!
//! [`fit_subblock_scale`] is the part of `quantize_row_iq4_nl_impl` that runs
//! BEFORE the branch on `super_block_size/block_size`, so IQ4_XS calls it once
//! per sub-block rather than duplicating it. `ntry = 7` (fixed, matching
//! `quantize_iq4_nl`'s call site) sweeps 15 candidate scales
//! `id = (itry + values[0]) / max` and keeps whichever maximises
//! `sumqx² / sumq2` — scored without the division, exactly as `ggml-quants.c`
//! computes it. The codebook lookup is NOT re-run against the winning scale
//! inside this loop; only `d` and `best` are updated. That is a deliberate
//! reproduction of the reference, not a bug: `ggml-quants.c` reuses whatever
//! `Lb[j]` step 1 assigned until the routine ends.
//!
//! # Final levels are re-derived, not carried from the search
//!
//! After the sweep picks `d`, every level is thrown away and recomputed once
//! more from `id = 1/d` — using the f32 `d`, not the f16 value written to the
//! block. Only THIS pass produces the levels that get packed. IQ4_XS instead
//! re-derives its levels against the QUANTIZED sub-scale — see its module docs.

use super::codebook::best_index_int8;
use super::search::{GROUP_MAX_EPS, block_sigma2, importance_weights};
use super::search_imatrix::block_importance;
use crate::quant::tables::KVALUES_IQ4NL;
use half::f16;

const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 18;
const NTRY: i32 = 7;

/// IQ4_NL, no importance weighting — llama.cpp `quantize_iq4_nl` with a null
/// `quant_weights`. Per-element weight is `x[j]²`.
///
/// Inverse of
/// [`dequant_iq4_nl`](crate::quant::cpu::kernels::dequant_iq4::dequant_iq4_nl).
pub fn quantize_iq4_nl(x: &[f32], out: &mut [u8]) {
    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut weight = [0.0f32; BLOCK_SIZE];
    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        for j in 0..BLOCK_SIZE {
            weight[j] = xb[j] * xb[j];
        }
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        quantize_block(xb, &weight, block);
    }
}

/// IQ4_NL weighted by an importance vector — llama.cpp `quantize_row_iq4_nl_impl`
/// with a non-null `quant_weights`. `imatrix` holds one entry per COLUMN, as in
/// the K-quant importance writers, and every row indexes the same vector.
///
/// Per-element weight is `qw[j] * sqrt(sigma2 + x[j]²)` with
/// `sigma2 = 2 * mean(x²)` over the block — [`block_sigma2`] returns the mean
/// alone, so the factor of 2 is applied here.
pub fn quantize_iq4_nl_imatrix(x: &[f32], out: &mut [u8], imatrix: &[f32]) {
    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut weight = [0.0f32; BLOCK_SIZE];
    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let qw = block_importance(imatrix, b, BLOCK_SIZE);
        let sigma2 = 2.0 * block_sigma2(xb);
        importance_weights(xb, qw, sigma2, &mut weight);

        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        quantize_block(xb, &weight, block);
    }
}

/// Per-sub-block scale search shared with IQ4_XS — the part of
/// `quantize_row_iq4_nl_impl` that runs BEFORE the branch on
/// `super_block_size/block_size` diverges the two formats. Returns the fitted
/// scale (`scales[ib]` in the source); a near-zero sub-block
/// (`amax < GROUP_MAX_EPS`) returns `0.0` without running the sweep.
///
/// `ntry` is always `7` from both current call sites, so the initial scale
/// always carries the sign opposite `max` — but the `ntry <= 0` branch is
/// ported too, matching `ggml-quants.c`'s
/// `d = ntry>0 ? -max/values[0] : max/values[0]` exactly.
pub(super) fn fit_subblock_scale(xb: &[f32], weight: &[f32], values: &[i8; 16], ntry: i32) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in xb {
        let av = v.abs();
        if av > amax {
            amax = av;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        return 0.0;
    }

    let mut id = if ntry > 0 {
        -values[0] as f32 / max
    } else {
        values[0] as f32 / max
    };
    let mut sumqx = 0.0f32;
    let mut sumq2 = 0.0f32;
    for (j, &xj) in xb.iter().enumerate() {
        let li = best_index_int8(values, id * xj);
        let q = values[li] as f32;
        let w = weight[j];
        sumqx += w * q * xj;
        sumq2 += w * q * q;
    }
    let mut d = sumqx / sumq2;
    let mut best = d * sumqx;

    for itry in -ntry..=ntry {
        id = (itry as f32 + values[0] as f32) / max;
        sumqx = 0.0;
        sumq2 = 0.0;
        for (j, &xj) in xb.iter().enumerate() {
            let li = best_index_int8(values, id * xj);
            let q = values[li] as f32;
            let w = weight[j];
            sumqx += w * q * xj;
            sumq2 += w * q * q;
        }
        // Deliberately NOT updating any level array here — `ggml-quants.c`
        // only refits the scale inside this loop and re-derives levels once,
        // after it, from whichever scale this loop settles on.
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }
    d
}

/// One 32-element block, weight already computed by either caller above.
///
/// A near-zero block (`amax < GROUP_MAX_EPS`) is NOT written as all-zero
/// bytes. `ggml-quants.c` sets `d = 0` and then still runs the final
/// re-derivation below with `id = 0`, which lands every nibble on codebook
/// index 8 (`KVALUES_IQ4NL[8] == 1`) — `0x88` repeated, not `0x00`. It
/// decodes back to exactly zero regardless, since the stored scale is zero.
fn quantize_block(xb: &[f32], weight: &[f32; BLOCK_SIZE], block: &mut [u8]) {
    let values = &KVALUES_IQ4NL;

    let d = fit_subblock_scale(xb, weight, values, NTRY);

    block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());

    // Re-derive every level from the f32 scale `d`, NOT the f16-rounded value
    // just written — `ggml-quants.c` reads `scales[0]` here, before rounding.
    // This is the ONLY pass whose levels get packed; steps above are search.
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut l = [0usize; BLOCK_SIZE];
    for j in 0..BLOCK_SIZE {
        l[j] = best_index_int8(values, id * xb[j]);
    }

    for j in 0..16 {
        block[2 + j] = (l[j] as u8) | ((l[j + 16] as u8) << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant_iq4::dequant_iq4_nl;

    /// A zero-input block stores a zero scale and decodes back to exact
    /// zero. The raw nibble bytes are `0x88` (codebook index 8), not `0x00`
    /// — see `quantize_block`'s doc comment for why that is not a bug.
    #[test]
    fn all_zero_block_writes_a_zero_scale_that_decodes_to_zero() {
        let x = [0.0f32; BLOCK_SIZE];
        let mut out = [0xFFu8; BLOCK_BYTES];
        quantize_iq4_nl(&x, &mut out);

        assert_eq!(f16::from_le_bytes([out[0], out[1]]).to_f32(), 0.0);

        let mut back = [1.0f32; BLOCK_SIZE];
        dequant_iq4_nl(&out, &mut back);
        for &v in &back {
            assert_eq!(v, 0.0);
        }
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.037 * (1.0 + (i / 32) as f32 * 0.1))
            .collect()
    }

    #[test]
    fn round_trips_through_existing_dequant_reader() {
        let x = ramp(64);
        let mut out = [0u8; 2 * BLOCK_BYTES];
        quantize_iq4_nl(&x, &mut out);

        let mut back = [0.0f32; 64];
        dequant_iq4_nl(&out, &mut back);

        for (got, want) in back.iter().zip(&x) {
            assert!((got - want).abs() < 0.5, "got {got}, want {want}");
        }
    }

    /// A spread input uses a spread of the codebook. Every 4-bit nibble is
    /// trivially a valid index, so that is not worth asserting; what can
    /// actually break is a writer collapsing onto one level — the shape a
    /// zeroed scale or a dead search produces.
    #[test]
    fn a_spread_input_reaches_many_codebook_entries() {
        let x = ramp(32);
        let mut out = [0u8; BLOCK_BYTES];
        quantize_iq4_nl(&x, &mut out);

        let mut seen = [false; 16];
        for &byte in &out[2..BLOCK_BYTES] {
            seen[usize::from(byte & 0x0F)] = true;
            seen[usize::from(byte >> 4)] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        assert!(
            distinct > 8,
            "a 32-value ramp reached only {distinct} levels"
        );
    }

    #[test]
    fn block_size_is_18_bytes_covering_32_elements() {
        assert_eq!(BLOCK_BYTES, 18);
        assert_eq!(BLOCK_SIZE, 32);
    }

    #[test]
    fn values_exactly_on_codebook_grid_reconstruct_near_exactly() {
        let d = 2.5f32;
        let x: Vec<f32> = (0..BLOCK_SIZE)
            .map(|j| d * KVALUES_IQ4NL[j % 16] as f32)
            .collect();

        let mut out = [0u8; BLOCK_BYTES];
        quantize_iq4_nl(&x, &mut out);

        let mut back = [0.0f32; BLOCK_SIZE];
        dequant_iq4_nl(&out, &mut back);

        for (got, want) in back.iter().zip(&x) {
            assert!((got - want).abs() < 0.05, "got {got}, want {want}");
        }
    }

    #[test]
    fn importance_weighted_path_runs_and_differs_from_unweighted() {
        // A skewed importance vector makes the weighted fit favor different
        // elements than plain x^2 weighting, on a block where the two
        // objectives disagree about which outlier to protect.
        let mut x = [0.3f32; BLOCK_SIZE];
        x[0] = 6.0;
        x[1] = -5.5;
        let mut imatrix = [1.0f32; BLOCK_SIZE];
        imatrix[1] = 50.0;

        let mut plain = [0u8; BLOCK_BYTES];
        quantize_iq4_nl(&x, &mut plain);

        let mut weighted = [0u8; BLOCK_BYTES];
        quantize_iq4_nl_imatrix(&x, &mut weighted, &imatrix);

        assert_ne!(plain, weighted);
    }
}
