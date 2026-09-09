//! Quantization writers for the simple 32-element block formats
//!
//! Q4_0, Q4_1, Q8_0. These have no sub-block structure. Their stored fields are
//! still free parameters: rounding the codes against a slightly different scale
//! can reconstruct the block more closely than the direct fit does. Q4_0 and
//! Q8_0 sweep their single scale through [`super::block_scale`]. Q4_1 sweeps its
//! scale AND offset through [`super::block_affine`], whose `m` is signed and
//! added, the sign that keeps it out of both other searches.
//!
//! # Codes are computed against the STORED fields
//!
//! Every stored field is binary16, so the reader reconstructs from the ROUNDED
//! values. Each code here divides by that same rounded scale, and Q4_1 subtracts
//! the same rounded offset, never the wider floats they came from. Q6_K applies
//! the same second-pass rule after quantizing its sub-block scales.
//!
//! # None of the three match llama.cpp byte for byte
//!
//! llama.cpp's `quantize_row_q4_0`, `quantize_row_q4_1` and
//! `quantize_row_q8_0` are plain direct fits with no search. Any block where a
//! search picks different parameters produces different bytes. The output stays
//! a VALID block — same size, layout and code range, decoded correctly by every
//! reader including llama.cpp's. It is a better encoding of the same format.
//! [`super::block_scale`] and [`super::block_affine`] state the full trade.
//!
//! # Nibble ordering — the trap
//!
//! Element `j` and element `j + 16` share ONE byte: low nibble is the first half
//! of the block, high nibble the second half. They are NOT adjacent output
//! positions. Packing `out[2i]`/`out[2i+1]` instead permutes every weight within
//! the block while keeping shape, block count and tensor RMS intact. That bug
//! shipped in compressr and failed no check until a model produced garbage.
//! `dequant_simple.rs` documents the same split for the read side and is the
//! authority here.

#[cfg(test)]
use super::block_affine::minmax_block_affine;
use super::block_affine::{BlockAffineFit, Q4_1_FIT, affine_code, fit_block_affine};
#[cfg(test)]
use super::block_scale::absmax_block_scale;
use super::block_scale::{BlockScaleFit, Q4_0_FIT, Q8_0_FIT, block_code, fit_block_scale};
use half::f16;

/// Per-block scale fit: `(values, format) -> stored binary16 scale`
type ScaleFit = fn(&[f32], &BlockScaleFit) -> f16;

/// Per-block affine fit: `(values, format) -> stored binary16 scale and offset`
type AffineFit = fn(&[f32], &BlockAffineFit) -> (f16, f16);

/// Q4_0: 32 elements, 18 bytes — 2-byte f16 `d`, then 16 nibble-pair bytes
///
/// Inverse of [`dequant_q4_0`](crate::quant::cpu::kernels::dequant_simple::dequant_q4_0),
/// which reads `d` at byte 0 and `qs` at bytes 2..18 with
/// `out[j] = (qs[j] & 0xF) - 8`, `out[j + 16] = (qs[j] >> 4) - 8`.
///
/// `d` carries the OPPOSITE sign to the largest-magnitude element in the block.
/// That puts that element on level -8, the one extra step the negative side has,
/// instead of wasting it.
pub fn quantize_q4_0(x: &[f32], out: &mut [u8]) {
    quantize_q4_0_with(x, out, fit_block_scale)
}

/// Q4_0 with an explicit scale fit — lets tests measure the search against absmax
pub(super) fn quantize_q4_0_with(x: &[f32], out: &mut [u8], fit: ScaleFit) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];

        let d = fit(xb, &Q4_0_FIT);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        let df = d.to_f32();

        for j in 0..16 {
            // The nibble stores the level biased by +8, matching the reader's
            // `- 8`. `block_code` clamps to [-8, 7] first.
            let lo = (block_code(xb[j], df, &Q4_0_FIT) + 8) as u8;
            let hi = (block_code(xb[j + 16], df, &Q4_0_FIT) + 8) as u8;
            block[2 + j] = lo | (hi << 4);
        }
    }
}

/// Q4_1: 32 elements, 20 bytes — f16 `d`, f16 `m`, then 16 nibble-pair bytes
///
/// Inverse of [`dequant_q4_1`](crate::quant::cpu::kernels::dequant_simple::dequant_q4_1),
/// which reads `d` at byte 0, `m` at byte 2, `qs` at bytes 4..20 and computes
/// `out = d·q + m`. Note the sign: Q4_1's min is ADDED, unlike Q4_K's `dmin`
/// which is subtracted. Both fields are binary16, so the search sweeps and
/// scores them exactly as the reader loads them.
pub fn quantize_q4_1(x: &[f32], out: &mut [u8]) {
    quantize_q4_1_with(x, out, fit_block_affine)
}

/// Q4_1 with an explicit affine fit — lets tests measure the search against min/max
pub(super) fn quantize_q4_1_with(x: &[f32], out: &mut [u8], fit: AffineFit) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];

        let (d, m) = fit(xb, &Q4_1_FIT);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&m.to_le_bytes());
        let (df, mf) = (d.to_f32(), m.to_f32());

        for j in 0..16 {
            // `affine_code` clamps to [0, 15], the unsigned nibble range the
            // reader uses directly with no bias to subtract.
            let lo = affine_code(xb[j], df, mf, &Q4_1_FIT) as u8;
            let hi = affine_code(xb[j + 16], df, mf, &Q4_1_FIT) as u8;
            block[4 + j] = lo | (hi << 4);
        }
    }
}

/// Q8_0: 32 elements, 34 bytes — 2-byte f16 `d`, then 32 signed bytes
///
/// Inverse of [`dequant_q8_0`](crate::quant::cpu::kernels::dequant_simple::dequant_q8_0),
/// which reads `d` at byte 0 and `qs` at bytes 2..34 with `out[i] = qs[i]·d`.
/// Unlike the 4-bit formats there is no split-half ordering: element `i` is
/// byte `i`. Codes are clamped to `[-127, 127]`, one short of the signed-byte
/// range, so the block stays symmetric.
pub fn quantize_q8_0(x: &[f32], out: &mut [u8]) {
    quantize_q8_0_with(x, out, fit_block_scale)
}

/// Q8_0 with an explicit scale fit — lets tests measure the search against absmax
pub(super) fn quantize_q8_0_with(x: &[f32], out: &mut [u8], fit: ScaleFit) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];

        let d = fit(xb, &Q8_0_FIT);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        let df = d.to_f32();

        for (j, &v) in xb.iter().enumerate() {
            block[2 + j] = block_code(v, df, &Q8_0_FIT) as i8 as u8;
        }
    }
}

/// Absmax baseline for Q4_0 — the scale choice the search replaces
///
/// Exists only so tests can assert the search beats it. Both paths share the
/// encoder, so the sweep is their only difference.
#[cfg(test)]
pub(super) fn quantize_q4_0_absmax(x: &[f32], out: &mut [u8]) {
    quantize_q4_0_with(x, out, absmax_block_scale)
}

/// Absmax baseline for Q8_0 — see [`quantize_q4_0_absmax`]
#[cfg(test)]
pub(super) fn quantize_q8_0_absmax(x: &[f32], out: &mut [u8]) {
    quantize_q8_0_with(x, out, absmax_block_scale)
}

/// Min/max baseline for Q4_1 — see [`quantize_q4_0_absmax`]
#[cfg(test)]
pub(super) fn quantize_q4_1_minmax(x: &[f32], out: &mut [u8]) {
    quantize_q4_1_with(x, out, minmax_block_affine)
}
