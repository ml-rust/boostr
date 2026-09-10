//! Q3_K importance-weighted writer — llama.cpp `quantize_row_q3_K_impl`
//!
//! The block layout, the two-pass requantization and the packing are the
//! no-imatrix path's, reused verbatim from [`super::q3k`]. Three things change:
//!
//! 1. **The sub-block search.** The `_ref` path runs `make_q3_quants`,
//!    coordinate descent over the levels with `w = x²`. The `_impl` path runs
//!    `make_qx_quants`, the scale sweep, with the explicit weight
//!    `w = qw · sqrt(sigma2 + x²)` and `sigma2 = 2·Σx²/n` over the whole
//!    super-block — [`super::search::importance_weights`]. Two different
//!    searches, not one search with two weights.
//! 2. **The super-block scale.** The `_ref` path divides the 16 sub-block
//!    scales by `-max_scale/32` and rounds. The `_impl` path runs the same
//!    `make_qx_quants` over those 16 scales, weighted by `sw[j]` — the sum of
//!    sub-block `j`'s own element weights — and stores the scale it returns.
//! 3. **What is stored.** `d` is the scale `make_qx_quants` returned, not
//!    `1/iscale`, and the 16 six-bit fields are the biased levels it wrote.
//!
//! The no-imatrix entry point is untouched and stays bit-identical to
//! llama.cpp's `_ref` output.

use super::q3k::{
    BLOCK_BYTES, NMAX, SCALE_BIAS, SUB, SUB_BLOCKS, SUPER_BLOCK, pack_q3k, pack_q3k_scale_levels,
};
use super::search::{MAX_SUB_BLOCK, block_sigma2, importance_weights, nearest_int};
use super::search_imatrix::{block_importance, make_qx_quants_weighted};
use crate::quant::cpu::kernels::dequant_k_quants::unpack_q3k_scales;
use half::f16;

/// Q3_K weighted by an importance vector — llama.cpp `quantize_row_q3_K_impl`
///
/// `imatrix` holds one non-negative entry per COLUMN of the weight matrix, so
/// its length is the row length and every row indexes the same vector.
///
/// Inverse of [`dequant_q3k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q3k),
/// exactly as [`quantize_q3k`](super::q3k::quantize_q3k) is.
pub fn quantize_q3k_imatrix(x: &[f32], out: &mut [u8], imatrix: &[f32]) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut sw = [0.0f32; SUB_BLOCKS];
    let mut weights = [0.0f32; MAX_SUB_BLOCK];

    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        block.fill(0);

        let qw = block_importance(imatrix, b, SUPER_BLOCK);
        // Q3_K, Q4_K and Q5_K take TWICE the super-block mean square, where
        // Q2_K takes it as it stands. `ggml-quants.c` is the authority.
        let sigma2 = 2.0 * block_sigma2(xb);
        for j in 0..SUB_BLOCKS {
            let xs = &xb[SUB * j..][..SUB];
            sw[j] = importance_weights(xs, &qw[SUB * j..][..SUB], sigma2, &mut weights[..SUB]);
            scales[j] =
                make_qx_quants_weighted(xs, NMAX, &mut levels[SUB * j..][..SUB], &weights[..SUB]);
        }

        // The 16 sub-block scales get the same symmetric search one level up,
        // weighted by how much each sub-block matters to the row. `scale_levels`
        // comes back BIASED by `SCALE_BIAS`, which is what the 6-bit fields
        // store, so nothing here re-derives them from a maximum.
        let mut scale_levels = [0u8; SUB_BLOCKS];
        let d_block = make_qx_quants_weighted(&scales, SCALE_BIAS, &mut scale_levels, &sw);
        pack_q3k_scale_levels(&scale_levels, &mut block[96..108]);
        let d = f16::from_f32(d_block);
        block[108..110].copy_from_slice(&d.to_le_bytes());

        // Second pass: levels against the scale the READER reconstructs, i.e.
        // the f16 `d` times the unpacked 6-bit scale, not the exact float from
        // the search. A sub-block whose stored scale rounded to zero keeps the
        // search's own levels, which the zero factor makes moot.
        let sc = unpack_q3k_scales(&block[96..108]);
        for j in 0..SUB_BLOCKS {
            let dl = d.to_f32() * sc[j] as f32;
            if dl == 0.0 {
                continue;
            }
            for ii in 0..SUB {
                let l = nearest_int(xb[SUB * j + ii] / dl).clamp(-NMAX, NMAX - 1);
                levels[SUB * j + ii] = (l + NMAX) as u8;
            }
        }

        pack_q3k(&levels, block);
    }
}
