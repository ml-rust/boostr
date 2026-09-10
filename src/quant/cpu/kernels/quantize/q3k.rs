//! Q3_K quantization writer
//!
//! 256-element super-block, 16 sub-blocks of 16. Each sub-block gets a signed
//! scale from the coordinate-descent search in
//! [`super::search::make_q3_quants`]; the 16 scales are then quantized to 6
//! bits each against one f16 super-block `d`. There is no min — Q3_K
//! reconstructs `x ≈ d · sc_j · (q − 4)` with `q ∈ [0, 7]`.
//!
//! # Field order
//!
//! `hmask`@0..32, `qs`@32..96, 12-byte `scales`@96..108, f16 `d`@108..110 —
//! 110 bytes. Like Q6_K, the f16 factor comes LAST, and writing GGML's fields
//! in declaration order from memory is not the same as writing them in this
//! order.
//! [`dequant_q3k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q3k)
//! is the authority for these offsets.
//!
//! # Split levels
//!
//! Each level is three bits, and the two planes are indexed differently. The
//! low two bits sit in `qs`, four elements to a byte at four shifts, 32 apart
//! — the same geometry Q2_K uses. The high bit sits in `hmask`, one BIT per
//! element: element `j` uses bit `j / 32` of `hmask[j % 32]`. The reader walks
//! that bit with a running mask that advances once per shift level and carries
//! across both 128-element halves, so the eight bits of a `hmask` byte cover
//! eight elements 32 apart.
//!
//! The reader SUBTRACTS 4 when the high bit is clear and 0 when it is set,
//! which is `q − 4` with `q`'s bit 2 taken from `hmask`. Storing the high bit
//! inverted still decodes, still round-trips inside a plausible error band on
//! symmetric data, and is wrong.
//!
//! # Two-pass requantization
//!
//! Levels are computed twice on purpose. The search picks levels against its
//! own exact float scale, but what the reader will actually see is the 6-bit
//! scale times the f16 `d`. The second pass re-derives every level against
//! THAT rounded scale, using the same unpacker the reader uses, so the two can
//! never drift apart.

#[cfg(test)]
use super::search::make_qx_absmax;
use super::search::{make_q3_quants, nearest_int};
use crate::quant::cpu::kernels::dequant_k_quants::unpack_q3k_scales;
use half::f16;

pub(super) const SUPER_BLOCK: usize = 256;
pub(super) const BLOCK_BYTES: usize = 110;
pub(super) const SUB_BLOCKS: usize = 16;
/// Elements per sub-block
pub(super) const SUB: usize = 16;
/// Q3_K levels span `[-4, 3]` and are stored biased by `+4`
pub(super) const NMAX: i32 = 4;
/// Stored sub-block scales span `[-32, 31]` and are stored biased by `+32`
pub(super) const SCALE_BIAS: i32 = 32;

/// Per-sub-block scale fit: `(values, nmax, biased_levels) -> scale`
type ScaleFit = fn(&[f32], i32, &mut [u8]) -> f32;

/// Q3_K: 256 elements, 110 bytes
///
/// Inverse of [`dequant_q3k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q3k).
pub fn quantize_q3k(x: &[f32], out: &mut [u8]) {
    quantize_q3k_with(x, out, make_q3_quants)
}

/// Q3_K with an explicit scale fit — lets tests measure the search against absmax
pub(super) fn quantize_q3k_with(x: &[f32], out: &mut [u8], fit: ScaleFit) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    let mut scales = [0.0f32; SUB_BLOCKS];

    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        block.fill(0);

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for j in 0..SUB_BLOCKS {
            let scale = fit(&xb[SUB * j..][..SUB], NMAX, &mut levels[SUB * j..][..SUB]);
            scales[j] = scale;
            if scale.abs() > max_abs_scale {
                max_abs_scale = scale.abs();
                max_scale = scale;
            }
        }

        // An all-zero super-block keeps the zeroed scales and a zero `d`.
        let d = if max_abs_scale != 0.0 {
            // Scales are SIGNED 6-bit. The negative divisor mirrors the
            // symmetric level range, so the sub-block carrying `max_scale`
            // lands on -32, which is 0 once biased.
            let iscale = -32.0 / max_scale;
            pack_q3k_scales(iscale, &scales, &mut block[96..108]);
            f16::from_f32(1.0 / iscale)
        } else {
            f16::from_f32(0.0)
        };
        block[108..110].copy_from_slice(&d.to_le_bytes());

        // Second pass: levels against the scale the READER reconstructs, i.e.
        // the f16 `d` times the unpacked 6-bit scale, not the exact float from
        // the fit. A sub-block whose stored scale rounded to zero keeps the
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

/// Quantize the 16 sub-block scales to signed 6 bits and pack them into 12 bytes
///
/// Bytes 0..8 carry one 4-bit nibble per sub-block: sub-block `j < 8` in the low
/// nibble of byte `j`, sub-block `j >= 8` in the high nibble of byte `j - 8`.
/// Bytes 8..12 carry the remaining top two bits: sub-block `j` at bit pair
/// `j / 4` of byte `8 + j % 4`. Mirrors
/// [`unpack_q3k_scales`](crate::quant::cpu::kernels::dequant_k_quants::unpack_q3k_scales)
/// exactly, which reassembles the same fields with 32-bit masks.
fn pack_q3k_scales(iscale: f32, scales: &[f32; SUB_BLOCKS], sc: &mut [u8]) {
    let mut biased = [0u8; SUB_BLOCKS];
    for j in 0..SUB_BLOCKS {
        biased[j] =
            (nearest_int(iscale * scales[j]).clamp(-SCALE_BIAS, SCALE_BIAS - 1) + SCALE_BIAS) as u8;
    }
    pack_q3k_scale_levels(&biased, sc);
}

/// Pack 16 ALREADY-BIASED 6-bit sub-block scales into 12 bytes
///
/// Split out because the importance path gets its biased levels straight from
/// [`super::search_imatrix::make_qx_quants_weighted`], which fits the 16 scales
/// against a weighted objective instead of dividing them by their maximum. The
/// bit layout is the same either way, and there is only one copy of it.
pub(super) fn pack_q3k_scale_levels(biased: &[u8; SUB_BLOCKS], sc: &mut [u8]) {
    for j in 0..SUB_BLOCKS {
        let l = biased[j];
        if j < 8 {
            sc[j] = l & 0x0F;
        } else {
            sc[j - 8] |= (l & 0x0F) << 4;
        }
        sc[8 + j % 4] |= (l >> 4) << (2 * (j / 4));
    }
}

/// Split the 3-bit levels into 2 low bits in `qs` and 1 high bit in `hmask`
///
/// `qs` byte `32n + l` for `l` in `0..32` carries elements `128n + l`,
/// `+32`, `+64` and `+96` at shifts 0, 2, 4 and 6. `hmask[j % 32]` bit
/// `j / 32` carries element `j`'s bit 2.
pub(super) fn pack_q3k(levels: &[u8; SUPER_BLOCK], block: &mut [u8]) {
    let mut low = [0u8; SUPER_BLOCK];
    for (j, &l) in levels.iter().enumerate() {
        if l > 3 {
            block[j % 32] |= 1u8 << (j / 32);
        }
        low[j] = l & 3;
    }
    for n in 0..2 {
        let base = n * 128;
        for l in 0..32 {
            block[32 + 32 * n + l] = low[base + l]
                | (low[base + l + 32] << 2)
                | (low[base + l + 64] << 4)
                | (low[base + l + 96] << 6);
        }
    }
}

/// Absmax baseline for Q3_K — the scale choice the search replaces.
///
/// This is llama.cpp's `make_q3_quants` with `do_rmse = false`: the plain
/// absmax scale with neither the least-squares correction nor the descent.
/// Exists only so the tests can assert the search actually beats it; nothing
/// in the shipped path should ever pick the worse scale on purpose.
#[cfg(test)]
pub(super) fn quantize_q3k_absmax(x: &[f32], out: &mut [u8]) {
    quantize_q3k_with(x, out, make_qx_absmax)
}
