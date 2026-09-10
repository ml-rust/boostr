//! Q2_K quantization writer
//!
//! 256-element super-block split into 16 sub-blocks of 16. Each sub-block gets
//! its own `(scale, min)` pair from the iterative search in
//! [`super::search::make_qkx2_quants`]; those 32 floats are then themselves
//! quantized to 4 bits each against two f16 super-block scales, `d` and `dmin`.
//!
//! # Reconstruction
//!
//! `x ≈ (d · scale_j) · q − (dmin · min_j)` with `q ∈ [0, 3]`. The min is
//! SUBTRACTED — that is why the search returns `-min` and why `min` is clamped
//! to ≤ 0 before it is stored unsigned. Q4_K and Q5_K use the same model with a
//! wider level grid and 6-bit stored scales.
//!
//! # Field order
//!
//! `scales`@0..16, `qs`@16..80, f16 `d`@80..82, f16 `dmin`@82..84 — 84 bytes.
//! Unlike Q4_K, the scales come FIRST and the two f16 factors come LAST.
//! [`dequant_q2k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q2k)
//! is the authority for these offsets.
//!
//! One `scales` byte serves one sub-block: low nibble is the 4-bit scale, high
//! nibble the 4-bit min. Sixteen bytes, sixteen sub-blocks, no interleave.
//!
//! # Absolute error, not squared
//!
//! Q2_K is the one format llama.cpp scores with `use_mad = true`. Two bits per
//! element leaves no room to buy an outlier: under the squared metric a single
//! extreme value drags the scale and the other fifteen elements pay for it.
//!
//! # Two-pass requantization
//!
//! Levels are computed twice on purpose. The search picks levels against its
//! own exact float scale, but what the reader will actually see is the 4-bit
//! scale times the f16 `d`. The second pass re-derives every level against
//! THAT rounded scale, reading the nibbles back out of the block the same way
//! the reader does, so the two can never drift apart.

use super::search::{KSearch, MAX_SUB_BLOCK, make_qkx2_quants, nearest_int};
use half::f16;

pub(super) const SUPER_BLOCK: usize = 256;
pub(super) const BLOCK_BYTES: usize = 84;
pub(super) const SUB_BLOCKS: usize = 16;
/// Elements per sub-block — one `scales` byte covers this many values
pub(super) const SUB: usize = 16;
/// Both the scale and the min are stored as a 4-bit fraction of an f16 factor
///
/// llama.cpp's `q4scale`.
const Q4SCALE: f32 = 15.0;

/// Q2_K search constants — `make_qkx2_quants(16, 3, ..., -0.5f, 0.1f, 15, true)`
pub(super) const Q2K_SEARCH: KSearch = KSearch {
    nmax: 3,
    rmin: -0.5,
    rdelta: 0.1,
    nstep: 15,
    use_mad: true,
};

/// Q2_K: 256 elements, 84 bytes
///
/// Inverse of [`dequant_q2k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q2k).
pub fn quantize_q2k(x: &[f32], out: &mut [u8]) {
    quantize_q2k_with(x, out, &Q2K_SEARCH)
}

/// Q2_K with explicit search parameters — lets tests measure search vs min/max
pub(super) fn quantize_q2k_with(x: &[f32], out: &mut [u8], search: &KSearch) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut mins = [0.0f32; SUB_BLOCKS];
    let mut weights = [0.0f32; MAX_SUB_BLOCK];
    let mut laux = [0u8; MAX_SUB_BLOCK];

    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        block.fill(0);

        // Both stored sets are non-negative: the scale because the level grid
        // is one-sided, the min because it is subtracted rather than added.
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..SUB_BLOCKS {
            let xs = &xb[SUB * j..][..SUB];
            // Weight = each element's own magnitude. Q4_K and Q5_K add the
            // sub-block RMS on top; Q2_K does not.
            for (w, &v) in weights.iter_mut().zip(xs) {
                *w = v.abs();
            }
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
            max_scale = max_scale.max(scale);
            max_min = max_min.max(min);
        }

        // An all-zero super-block keeps the zeroed scales and a zero factor.
        let d = if max_scale > 0.0 {
            let iscale = Q4SCALE / max_scale;
            for j in 0..SUB_BLOCKS {
                block[j] = nearest_int(iscale * scales[j]).clamp(0, 15) as u8;
            }
            f16::from_f32(max_scale / Q4SCALE)
        } else {
            f16::from_f32(0.0)
        };
        let dmin = if max_min > 0.0 {
            let iscale = Q4SCALE / max_min;
            for j in 0..SUB_BLOCKS {
                block[j] |= (nearest_int(iscale * mins[j]).clamp(0, 15) as u8) << 4;
            }
            f16::from_f32(max_min / Q4SCALE)
        } else {
            f16::from_f32(0.0)
        };
        block[80..82].copy_from_slice(&d.to_le_bytes());
        block[82..84].copy_from_slice(&dmin.to_le_bytes());

        // Second pass: levels against the scale the READER reconstructs, i.e.
        // the nibbles now in `block`, not the exact floats from the search. A
        // sub-block whose stored scale rounded to zero keeps the search's own
        // levels, which the zero factor makes moot.
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

/// Pack the 2-bit levels into `qs`
///
/// Per 128-element half `n`, byte `32n + l` for `l` in `0..32` carries four
/// elements at four shifts: `128n + l` at bits 0-1, `+32` at 2-3, `+64` at 4-5
/// and `+96` at 6-7. The reader walks the same halves and shifts, so the four
/// elements sharing a byte sit 32 apart, never adjacent.
pub(super) fn pack_q2k(levels: &[u8; SUPER_BLOCK], qs: &mut [u8]) {
    for n in 0..2 {
        let base = n * 128;
        for l in 0..32 {
            qs[32 * n + l] = levels[base + l]
                | (levels[base + l + 32] << 2)
                | (levels[base + l + 64] << 4)
                | (levels[base + l + 96] << 6);
        }
    }
}
