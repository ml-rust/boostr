//! Q4_K and Q5_K importance-weighted writers — llama.cpp
//! `quantize_row_q4_K_impl` and `quantize_row_q5_K_impl`
//!
//! The block layout, the two-pass requantization and the packing are the
//! no-imatrix path's, reused verbatim from [`super::q4k_q5k`]. Three things
//! change, all of them in how the scales are chosen:
//!
//! 1. **The sub-block weight.** The `_ref` path uses `w = sqrt(Σx²/n) + |x|`,
//!    which knows only the data. The `_impl` path uses
//!    `w = qw · sqrt(sigma2 + x²)` with `sigma2 = 2·Σx²/n` over the whole
//!    super-block — [`super::search::importance_weights`].
//! 2. **The sweep.** `make_qkx3_quants(..., -0.9, 0.05, 36, false)` replaces
//!    `make_qkx2_quants(..., -1.0, 0.1, 20, false)` for Q4_K and
//!    `(..., -0.5, 0.1, 15, false)` for Q5_K. The two C routines are the same
//!    routine for these call sites — [`super::search_imatrix`] proves it — so
//!    only the constants differ here.
//! 3. **The super-block factor.** The `_ref` path divides the 8 sub-block
//!    scales by their maximum and rounds. The `_impl` path fits them with
//!    [`super::search_imatrix::make_qp_quants`] weighted by `sw[j]`, the sum of
//!    sub-block `j`'s own element weights, so the sub-blocks the row's
//!    activations actually depend on keep their scales.
//!
//! The no-imatrix entry points are untouched and stay bit-identical to
//! llama.cpp's `_ref` output.

use super::q4k_q5k::{
    Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES, SUB_BLOCKS, SUPER_BLOCK, pack_q4k, pack_q5k, pack_scale_mins,
    requantize, write_factors,
};
use super::search::{KSearch, MAX_SUB_BLOCK, block_sigma2, importance_weights, make_qkx2_quants};
use super::search_imatrix::{block_importance, make_qp_quants};
use half::f16;

/// Elements per sub-block
const SUB: usize = 32;
/// Stored sub-block scales and mins are 6-bit fractions of an f16 factor
const NMAX_STORED: i32 = 63;

/// Q4_K importance search constants — `make_qkx3_quants(32, 15, ..., -0.9f, 0.05f, 36, false)`
const Q4K_IMATRIX_SEARCH: KSearch = KSearch {
    nmax: 15,
    rmin: -0.9,
    rdelta: 0.05,
    nstep: 36,
    use_mad: false,
};

/// Q5_K importance search constants — `make_qkx3_quants(32, 31, ..., -0.9f, 0.05f, 36, false)`
const Q5K_IMATRIX_SEARCH: KSearch = KSearch {
    nmax: 31,
    rmin: -0.9,
    rdelta: 0.05,
    nstep: 36,
    use_mad: false,
};

/// Q4_K weighted by an importance vector — llama.cpp `quantize_row_q4_K_impl`
///
/// `imatrix` holds one non-negative entry per COLUMN of the weight matrix, so
/// its length is the row length and every row indexes the same vector.
pub fn quantize_q4k_imatrix(x: &[f32], out: &mut [u8], imatrix: &[f32]) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * Q4K_BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * Q4K_BLOCK_BYTES..][..Q4K_BLOCK_BYTES];
        block.fill(0);

        let mut sc = [0u8; 12];
        let qw = block_importance(imatrix, b, SUPER_BLOCK);
        let (d, dmin) =
            fit_super_block_imatrix(xb, qw, &Q4K_IMATRIX_SEARCH, false, &mut levels, &mut sc);
        write_factors(d, dmin, &sc, block);
        pack_q4k(&levels, block);
    }
}

/// Q5_K weighted by an importance vector — llama.cpp `quantize_row_q5_K_impl`
///
/// `imatrix` holds one non-negative entry per COLUMN of the weight matrix, so
/// its length is the row length and every row indexes the same vector.
pub fn quantize_q5k_imatrix(x: &[f32], out: &mut [u8], imatrix: &[f32]) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * Q5K_BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * Q5K_BLOCK_BYTES..][..Q5K_BLOCK_BYTES];
        block.fill(0);

        let mut sc = [0u8; 12];
        let qw = block_importance(imatrix, b, SUPER_BLOCK);
        let (d, dmin) =
            fit_super_block_imatrix(xb, qw, &Q5K_IMATRIX_SEARCH, true, &mut levels, &mut sc);
        write_factors(d, dmin, &sc, block);
        pack_q5k(&levels, block);
    }
}

/// Search every sub-block against the importance, fit the 16 resulting
/// scales/mins to 6 bits, then re-derive the element levels
///
/// Returns the f16-rounded `(d, dmin)` and fills `sc` with the 12-byte packed
/// scale/min array the reader's `unpack_q4k_q5k_scales` expects.
///
/// `clamp_stored` reproduces a difference the two C routines really do have:
/// `quantize_row_q5_K_impl` runs `ls = MIN(63, ls)` on the levels
/// `make_qp_quants` returned, and `quantize_row_q4_K_impl` does not. It is
/// unreachable on any scale or min those searches produce — both are
/// non-negative and `make_qp_quants` caps its final assignment at `nmax` — but
/// the two writers are not the same writer, and a byte-equality gate is not the
/// place to assume a line is dead.
fn fit_super_block_imatrix(
    x: &[f32],
    qw: &[f32],
    search: &KSearch,
    clamp_stored: bool,
    levels: &mut [u8; SUPER_BLOCK],
    sc: &mut [u8; 12],
) -> (f32, f32) {
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut mins = [0.0f32; SUB_BLOCKS];
    let mut sw = [0.0f32; SUB_BLOCKS];
    let mut weights = [0.0f32; MAX_SUB_BLOCK];
    let mut laux = [0u8; MAX_SUB_BLOCK];

    // Q4_K and Q5_K take TWICE the super-block mean square, where Q2_K takes it
    // as it stands. `ggml-quants.c` is the authority for the factor.
    let sigma2 = 2.0 * block_sigma2(x);

    for j in 0..SUB_BLOCKS {
        let xs = &x[SUB * j..][..SUB];
        sw[j] = importance_weights(xs, &qw[SUB * j..][..SUB], sigma2, &mut weights);
        let (scale, min) = make_qkx2_quants(
            xs,
            search.nmax,
            &weights,
            &mut levels[SUB * j..][..SUB],
            &mut laux,
            search.rmin,
            search.rdelta,
            search.nstep,
            search.use_mad,
        );
        scales[j] = scale;
        mins[j] = min;
    }

    // The 8 scales and the 8 mins each get their own weighted one-sided fit
    // against the shared `sw`, rather than a division by their maximum.
    let mut ls = [0u8; SUB_BLOCKS];
    let mut lm = [0u8; SUB_BLOCKS];
    let d_block = make_qp_quants(&scales, NMAX_STORED, &mut ls, &sw);
    let m_block = make_qp_quants(&mins, NMAX_STORED, &mut lm, &sw);
    if clamp_stored {
        for j in 0..SUB_BLOCKS {
            ls[j] = ls[j].min(NMAX_STORED as u8);
            lm[j] = lm[j].min(NMAX_STORED as u8);
        }
    }
    pack_scale_mins(&ls, &lm, sc);

    let d = f16::from_f32(d_block).to_f32();
    let dmin = f16::from_f32(m_block).to_f32();
    requantize(x, d, dmin, sc, search.nmax, levels);

    (d, dmin)
}
