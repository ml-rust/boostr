//! Q4_K and Q5_K quantization writers
//!
//! Both are 256-element super-blocks split into 8 sub-blocks of 32. Each
//! sub-block gets its own `(scale, min)` pair from the iterative search in
//! [`super::search::make_qkx2_quants`]; those 16 floats are then themselves
//! quantized to 6 bits each against two f16 super-block scales, `d` and `dmin`.
//!
//! # Reconstruction
//!
//! `x ≈ (d · scale_j) · q − (dmin · min_j)`. The min is SUBTRACTED — that is
//! why the search returns `-min` and why `min` is clamped to ≤ 0 before it is
//! stored unsigned.
//!
//! # Two-pass requantization
//!
//! Levels are computed twice on purpose. The search picks levels against its
//! own exact float scale, but what the reader will actually see is the 6-bit
//! scale times the f16 `d`. The second pass re-derives every level against
//! THAT rounded scale, using the same unpacker the reader uses, so the two can
//! never drift apart.

use super::search::{KSearch, MAX_SUB_BLOCK, make_qkx2_quants, nearest_int};
use crate::quant::cpu::kernels::dequant_k_quants::unpack_q4k_q5k_scales;
use half::f16;

pub(super) const SUPER_BLOCK: usize = 256;
pub(super) const SUB_BLOCKS: usize = 8;
/// Q4_K: f16 `d`@0, f16 `dmin`@2, 12-byte scales@4, 128-byte `qs`@16
pub(super) const Q4K_BLOCK_BYTES: usize = 144;
/// Q5_K: f16 `d`@0, f16 `dmin`@2, scales@4, 32-byte `qh`@16, 128-byte `qs`@48
pub(super) const Q5K_BLOCK_BYTES: usize = 176;

/// Q4_K search constants — `make_qkx2_quants(32, 15, ..., -1.0f, 0.1f, 20, false)`
pub(super) const Q4K_SEARCH: KSearch = KSearch {
    nmax: 15,
    rmin: -1.0,
    rdelta: 0.1,
    nstep: 20,
    use_mad: false,
};

/// Q5_K search constants — `make_qkx2_quants(32, 31, ..., -0.5f, 0.1f, 15, false)`
pub(super) const Q5K_SEARCH: KSearch = KSearch {
    nmax: 31,
    rmin: -0.5,
    rdelta: 0.1,
    nstep: 15,
    use_mad: false,
};

/// Q4_K: 256 elements, 144 bytes — f16 `d`@0, f16 `dmin`@2, 12-byte scales@4, 128-byte `qs`@16
///
/// Inverse of [`dequant_q4k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q4k).
/// Sub-block PAIRS share one 32-byte run of `qs`: the even sub-block owns the
/// low nibbles, the odd one the high nibbles of the SAME bytes.
pub fn quantize_q4k(x: &[f32], out: &mut [u8]) {
    quantize_q4k_with(x, out, &Q4K_SEARCH)
}

/// Q5_K: 256 elements, 176 bytes — f16 `d`@0, f16 `dmin`@2, scales@4, `qh`@16, `qs`@48
///
/// Inverse of [`dequant_q5k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q5k).
/// The low nibbles follow Q4_K exactly. The fifth bit lives in `qh`, indexed by
/// ELEMENT within the sub-block, with the BIT position being the sub-block
/// index — `qh[l] >> j`. It is not a flat bitstream over the 256 values;
/// getting it wrong scored cosine 0.03 against a reference build.
pub fn quantize_q5k(x: &[f32], out: &mut [u8]) {
    quantize_q5k_with(x, out, &Q5K_SEARCH)
}

/// Q4_K with explicit search parameters — lets tests measure search vs absmax
pub(super) fn quantize_q4k_with(x: &[f32], out: &mut [u8], search: &KSearch) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * Q4K_BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * Q4K_BLOCK_BYTES..][..Q4K_BLOCK_BYTES];
        block.fill(0);

        let mut sc = [0u8; 12];
        let (d, dmin) = fit_super_block(xb, search, &mut levels, &mut sc);
        write_factors(d, dmin, &sc, block);
        pack_q4k(&levels, block);
    }
}

/// Q5_K with explicit search parameters — lets tests measure search vs absmax
pub(super) fn quantize_q5k_with(x: &[f32], out: &mut [u8], search: &KSearch) {
    let num_blocks = x.len() / SUPER_BLOCK;
    debug_assert_eq!(out.len(), num_blocks * Q5K_BLOCK_BYTES);

    let mut levels = [0u8; SUPER_BLOCK];
    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK..][..SUPER_BLOCK];
        let block = &mut out[b * Q5K_BLOCK_BYTES..][..Q5K_BLOCK_BYTES];
        block.fill(0);

        let mut sc = [0u8; 12];
        let (d, dmin) = fit_super_block(xb, search, &mut levels, &mut sc);
        write_factors(d, dmin, &sc, block);
        pack_q5k(&levels, block);
    }
}

/// Write the two f16 factors and the 12-byte scale/min array
///
/// Q4_K and Q5_K share the first 16 bytes of a super-block exactly; only what
/// follows differs.
pub(super) fn write_factors(d: f32, dmin: f32, sc: &[u8; 12], block: &mut [u8]) {
    block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(dmin).to_le_bytes());
    block[4..16].copy_from_slice(sc);
}

/// Pack the 4-bit levels into Q4_K's `qs`
///
/// Sub-blocks 2k and 2k+1 share bytes `32k..32k+32`: the even sub-block owns
/// the low nibbles, the odd one the high nibbles of the SAME bytes.
pub(super) fn pack_q4k(levels: &[u8; SUPER_BLOCK], block: &mut [u8]) {
    for k in 0..SUB_BLOCKS / 2 {
        for l in 0..32 {
            let lo = levels[64 * k + l] & 0x0F;
            let hi = levels[64 * k + 32 + l] & 0x0F;
            block[16 + 32 * k + l] = lo | (hi << 4);
        }
    }
}

/// Pack the 5-bit levels into Q5_K's `qh` and `qs`
///
/// The low nibbles follow Q4_K exactly. The fifth bit goes to `qh`, indexed by
/// ELEMENT within the sub-block, with the BIT position being the sub-block
/// index. `block` must already be zeroed — `qh` is built by OR.
pub(super) fn pack_q5k(levels: &[u8; SUPER_BLOCK], block: &mut [u8]) {
    for k in 0..SUB_BLOCKS / 2 {
        for l in 0..32 {
            let mut l1 = levels[64 * k + l];
            let mut l2 = levels[64 * k + 32 + l];
            if l1 > 15 {
                l1 -= 16;
                block[16 + l] |= 1 << (2 * k);
            }
            if l2 > 15 {
                l2 -= 16;
                block[16 + l] |= 1 << (2 * k + 1);
            }
            block[48 + 32 * k + l] = l1 | (l2 << 4);
        }
    }
}

/// Search every sub-block, quantize the 16 resulting scales/mins to 6 bits,
/// then re-derive the element levels against those rounded scales
///
/// Returns the f16-rounded `(d, dmin)` and fills `sc` with the 12-byte packed
/// scale/min array the reader's `unpack_q4k_q5k_scales` expects.
fn fit_super_block(
    x: &[f32],
    search: &KSearch,
    levels: &mut [u8; SUPER_BLOCK],
    sc: &mut [u8; 12],
) -> (f32, f32) {
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut mins = [0.0f32; SUB_BLOCKS];
    let mut weights = [0.0f32; MAX_SUB_BLOCK];
    let mut laux = [0u8; MAX_SUB_BLOCK];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;

    for j in 0..SUB_BLOCKS {
        let xs = &x[32 * j..][..32];
        // Weight = RMS of the sub-block plus each element's own magnitude, so a
        // small element still gets a vote instead of being written off.
        let sum_x2: f32 = xs.iter().map(|v| v * v).sum();
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        let (scale, min) = make_qkx2_quants(
            xs,
            search.nmax,
            &weights,
            &mut levels[32 * j..][..32],
            &mut laux,
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

    // Both scale sets are stored as 6-bit fractions of a shared f16 factor.
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut ls = [0u8; SUB_BLOCKS];
    let mut lm = [0u8; SUB_BLOCKS];
    for j in 0..SUB_BLOCKS {
        ls[j] = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
        lm[j] = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
    }
    pack_scale_mins(&ls, &lm, sc);

    let d = f16::from_f32(max_scale / 63.0).to_f32();
    let dmin = f16::from_f32(max_min / 63.0).to_f32();
    requantize(x, d, dmin, sc, search.nmax, levels);

    (d, dmin)
}

/// Pack the 8 six-bit scales and 8 six-bit mins into the 12-byte `scales` array
///
/// The low four of each set are plain 6-bit values in `sc[0..4]` / `sc[4..8]`.
/// The high four put their bottom 4 bits in `sc[8..12]` and their top 2 bits in
/// the spare high bits of `sc[j-4]` (scales) and `sc[j]` (mins). Mirrors
/// [`unpack_q4k_q5k_scales`](crate::quant::cpu::kernels::dequant_k_quants::unpack_q4k_q5k_scales)
/// exactly. `sc` must arrive zeroed — the high four OR into bytes the low four
/// already wrote.
pub(super) fn pack_scale_mins(ls: &[u8; SUB_BLOCKS], lm: &[u8; SUB_BLOCKS], sc: &mut [u8; 12]) {
    for j in 0..SUB_BLOCKS {
        if j < 4 {
            sc[j] = ls[j];
            sc[j + 4] = lm[j];
        } else {
            sc[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
            sc[j - 4] |= (ls[j] >> 4) << 6;
            sc[j] |= (lm[j] >> 4) << 6;
        }
    }
}

/// Second pass: re-derive every level against the scales the READER will
/// reconstruct
///
/// The search picks levels against its own exact float scale; what the reader
/// sees is the 6-bit scale times the f16 `d`. Reading them back through the
/// reader's own unpacker is what keeps the two from drifting apart. A sub-block
/// whose stored scale rounds to zero keeps the search's levels, which the zero
/// factor makes moot.
pub(super) fn requantize(
    x: &[f32],
    d: f32,
    dmin: f32,
    sc: &[u8; 12],
    nmax: i32,
    levels: &mut [u8; SUPER_BLOCK],
) {
    let (sc_u, m_u) = unpack_q4k_q5k_scales(sc);
    for j in 0..SUB_BLOCKS {
        let dl = d * sc_u[j] as f32;
        if dl == 0.0 {
            continue;
        }
        let ml = dmin * m_u[j] as f32;
        for ii in 0..32 {
            let l = nearest_int((x[32 * j + ii] + ml) / dl).clamp(0, nmax);
            levels[32 * j + ii] = l as u8;
        }
    }
}
