//! Q2_K importance-weighted writer — llama.cpp `quantize_row_q2_K_impl`
//!
//! The block layout, the two-pass requantization and the packing are the
//! no-imatrix path's, reused verbatim from [`super::q2k`]. Three things change:
//!
//! 1. **The sub-block weight.** The `_ref` path uses `w = |x|`. The `_impl`
//!    path uses `w = qw · sqrt(sigma2 + x²)` with `sigma2 = Σx²/n` over the
//!    whole super-block — [`super::search::importance_weights`]. Note the
//!    factor: Q2_K takes the super-block mean square as it stands, where Q3_K,
//!    Q4_K and Q5_K double it.
//! 2. **The sweep, and its error metric.** `make_qkx3_quants(..., -0.9, 0.05,
//!    36, false)` replaces `make_qkx2_quants(..., -0.5, 0.1, 15, true)`. Q2_K
//!    is the one format whose `_ref` path scores candidates by ABSOLUTE error;
//!    its `_impl` path scores by squared error like every other format, because
//!    the importance already says which elements may be disappointed. The two C
//!    routines are the same routine for these call sites —
//!    [`super::search_imatrix`] proves it — so only the constants differ here.
//! 3. **The super-block factors.** The `_ref` path divides the 16 sub-block
//!    scales by their maximum and rounds. The `_impl` path fits them with
//!    [`super::search_imatrix::make_qp_quants`] weighted by `sw[j]`, the sum of
//!    sub-block `j`'s own element weights.
//!
//! The no-imatrix entry point is untouched and stays bit-identical to
//! llama.cpp's `_ref` output.

use super::q2k::{BLOCK_BYTES, SUB, SUB_BLOCKS, SUPER_BLOCK, pack_q2k};
use super::search::{
    KSearch, MAX_SUB_BLOCK, block_sigma2, importance_weights, make_qkx2_quants, nearest_int,
};
use super::search_imatrix::{block_importance, make_qp_quants};
use half::f16;

/// Both the scale and the min are stored as a 4-bit fraction of an f16 factor
const NMAX_STORED: i32 = 15;

/// Q2_K importance search constants — `make_qkx3_quants(16, 3, ..., -0.9f, 0.05f, 36, false)`
const Q2K_IMATRIX_SEARCH: KSearch = KSearch {
    nmax: 3,
    rmin: -0.9,
    rdelta: 0.05,
    nstep: 36,
    use_mad: false,
};

/// Q2_K weighted by an importance vector — llama.cpp `quantize_row_q2_K_impl`
///
/// `imatrix` holds one non-negative entry per COLUMN of the weight matrix, so
/// its length is the row length and every row indexes the same vector.
///
/// Inverse of [`dequant_q2k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q2k),
/// exactly as [`quantize_q2k`](super::q2k::quantize_q2k) is.
pub fn quantize_q2k_imatrix(x: &[f32], out: &mut [u8], imatrix: &[f32]) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let search = &Q2K_IMATRIX_SEARCH;
    let mut levels = [0u8; SUPER_BLOCK];
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut mins = [0.0f32; SUB_BLOCKS];
    let mut sw = [0.0f32; SUB_BLOCKS];
    let mut weights = [0.0f32; MAX_SUB_BLOCK];
    let mut laux = [0u8; MAX_SUB_BLOCK];

    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        block.fill(0);

        let qw = block_importance(imatrix, b, SUPER_BLOCK);
        let sigma2 = block_sigma2(xb);
        for j in 0..SUB_BLOCKS {
            let xs = &xb[SUB * j..][..SUB];
            // `ggml-quants.c` sums `QK_K/16` weights into `sw[j]` where the
            // sub-block holds 16; for the only super-block size GGUF defines
            // the two counts are the same 16, so this is that sum.
            sw[j] = importance_weights(xs, &qw[SUB * j..][..SUB], sigma2, &mut weights[..SUB]);
            let (scale, min) = make_qkx2_quants(
                xs,
                search.nmax,
                &weights[..SUB],
                &mut levels[SUB * j..][..SUB],
                &mut laux[..SUB],
                search.rmin,
                search.rdelta,
                search.nstep,
                search.use_mad,
            );
            scales[j] = scale;
            mins[j] = min;
        }

        // The 16 scales and the 16 mins each get their own weighted one-sided
        // fit against the shared `sw`, rather than a division by their maximum.
        let mut ls = [0u8; SUB_BLOCKS];
        let mut lm = [0u8; SUB_BLOCKS];
        let dm = make_qp_quants(&scales, NMAX_STORED, &mut ls, &sw);
        let mm = make_qp_quants(&mins, NMAX_STORED, &mut lm, &sw);
        let d = f16::from_f32(dm);
        let dmin = f16::from_f32(mm);
        block[80..82].copy_from_slice(&d.to_le_bytes());
        block[82..84].copy_from_slice(&dmin.to_le_bytes());
        for j in 0..SUB_BLOCKS {
            // One byte per sub-block: 4-bit scale low, 4-bit min high.
            block[j] = ls[j] | (lm[j] << 4);
        }

        // Second pass: levels against the scale the READER reconstructs, i.e.
        // the nibbles now in `block` times the f16 factors, not the exact
        // floats from the search. A sub-block whose stored scale rounded to
        // zero keeps the search's own levels, which the zero factor makes moot.
        for j in 0..SUB_BLOCKS {
            let dl = d.to_f32() * (block[j] & 0x0F) as f32;
            if dl == 0.0 {
                continue;
            }
            let ml = dmin.to_f32() * (block[j] >> 4) as f32;
            for ii in 0..SUB {
                let l = nearest_int((xb[SUB * j + ii] + ml) / dl).clamp(0, search.nmax);
                levels[SUB * j + ii] = l as u8;
            }
        }

        pack_q2k(&levels, &mut block[16..80]);
    }
}
