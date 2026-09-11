//! IQ4_XS quantization writer — llama.cpp `quantize_iq4_xs`
//!
//! 256 elements, 136 bytes: f16 super-scale `d`, `scales_h` (u16), `scales_l`
//! (4 bytes), then 128 bytes of nibble pairs — 8 sub-blocks of 32 elements.
//! Same codebook and the same per-sub-block scale fit as IQ4_NL
//! ([`super::iq4_nl::fit_subblock_scale`]), but IQ4_XS is the format that
//! actually exercises the `super_block_size/block_size > 1` branch of
//! `quantize_row_iq4_nl_impl` — IQ4_NL's single 32-element block never reaches
//! it. That branch requantizes the 8 fitted sub-block scales to 6 bits under
//! one f16 super-scale.
//!
//! # 6-bit sub-scale packing
//!
//! Each sub-block's biased level `l' = l + 32` (`l` the signed 6-bit value,
//! clamped to `[-32, 31]`) splits as `l_l = l' & 0xF`, `l_h = l' >> 4`.
//! `l_l` packs into `scales_l`, one nibble per sub-block — low nibble for even
//! `ib`, high nibble for odd `ib` — and `l_h` packs into the single
//! `scales_h` `u16`, 2 bits per sub-block at `l_h << (2*ib)`, covering ALL
//! EIGHT sub-blocks (16 bits = 8x2), not just four. Recovery:
//! `ls = l_l | (l_h << 4)`, `dl = d * (ls - 32)` — `dequant_iq4_xs`'s exact
//! convention, verified against this writer by the round-trip test below.
//!
//! # Final levels are derived against the QUANTIZED sub-scale
//!
//! Unlike IQ4_NL, which re-derives its 32 levels from the fitted float scale
//! directly, IQ4_XS re-derives each sub-block's 32 levels from
//! `dl = super_scale * quantized_6bit_l` — the ROUNDED sub-scale that
//! actually gets stored, not the float scale the search fit. This happens for
//! every sub-block unconditionally, including a near-zero one
//! (`fit_subblock_scale` returned `0.0`): its quantized `l` still rounds to
//! `0`, `dl` is `0`, and `best_index_int8` maps every zero-scaled element to
//! codebook index 8 — see `iq4_nl.rs`'s degenerate-block doc for why that is
//! not a bug.

use super::codebook::best_index_int8;
use super::iq4_nl::fit_subblock_scale;
use super::search::{block_sigma2, importance_weights, nearest_int};
use super::search_imatrix::block_importance;
use crate::quant::tables::KVALUES_IQ4NL;
use half::f16;

const SUPER_BLOCK_SIZE: usize = 256;
const SUB_BLOCK_SIZE: usize = 32;
const NUM_SUB_BLOCKS: usize = SUPER_BLOCK_SIZE / SUB_BLOCK_SIZE;
const BLOCK_BYTES: usize = 136;
const NTRY: i32 = 7;

/// IQ4_XS, no importance weighting — llama.cpp `quantize_iq4_xs` with a null
/// `quant_weights`. Per-element weight is `x[j]²`.
///
/// Inverse of
/// [`dequant_iq4_xs`](crate::quant::cpu::kernels::dequant_iq4::dequant_iq4_xs).
pub fn quantize_iq4_xs(x: &[f32], out: &mut [u8]) {
    let num_blocks = x.len() / SUPER_BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut weight = [0.0f32; SUB_BLOCK_SIZE];
    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK_SIZE..][..SUPER_BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        quantize_super_block(xb, &mut weight, block, None);
    }
}

/// IQ4_XS weighted by an importance vector — llama.cpp `quantize_iq4_xs` with
/// a non-null `quant_weights`. `imatrix` holds one entry per COLUMN, as in the
/// K-quant importance writers, and every row indexes the same vector.
///
/// Per-element weight is `qw[j] * sqrt(sigma2 + x[j]²)` with
/// `sigma2 = 2 * mean(x²)` taken over the WHOLE 256-element super-block, not
/// per sub-block — matching `quantize_row_iq4_nl_impl`'s single `sigma2`
/// computed once at the top, before the per-sub-block loop.
pub fn quantize_iq4_xs_imatrix(x: &[f32], out: &mut [u8], imatrix: &[f32]) {
    let num_blocks = x.len() / SUPER_BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    let mut weight = [0.0f32; SUB_BLOCK_SIZE];
    for b in 0..num_blocks {
        let xb = &x[b * SUPER_BLOCK_SIZE..][..SUPER_BLOCK_SIZE];
        let qw = block_importance(imatrix, b, SUPER_BLOCK_SIZE);
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];
        quantize_super_block(xb, &mut weight, block, Some(qw));
    }
}

/// One 256-element super-block. `weight` is scratch reused per sub-block, not
/// caller state.
fn quantize_super_block(
    xb: &[f32],
    weight: &mut [f32; SUB_BLOCK_SIZE],
    block: &mut [u8],
    quant_weights: Option<&[f32]>,
) {
    let values = &KVALUES_IQ4NL;
    let sigma2 = 2.0 * block_sigma2(xb);

    // Step 1: fit one float scale per sub-block — identical to IQ4_NL's own
    // per-block fit, just run 8 times.
    let mut scales = [0.0f32; NUM_SUB_BLOCKS];
    for ib in 0..NUM_SUB_BLOCKS {
        let sub = &xb[ib * SUB_BLOCK_SIZE..][..SUB_BLOCK_SIZE];
        match quant_weights {
            Some(qw) => {
                let sub_qw = &qw[ib * SUB_BLOCK_SIZE..][..SUB_BLOCK_SIZE];
                importance_weights(sub, sub_qw, sigma2, weight);
            }
            None => {
                for (j, &v) in sub.iter().enumerate() {
                    weight[j] = v * v;
                }
            }
        }
        scales[ib] = fit_subblock_scale(sub, weight, values, NTRY);
    }

    // Step 2: super-scale is `-max_scale/32`, where `max_scale` is the
    // sub-block scale with the largest MAGNITUDE, kept SIGNED.
    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;
    for &s in &scales {
        let a = s.abs();
        if a > amax_scale {
            amax_scale = a;
            max_scale = s;
        }
    }
    let d = -max_scale / 32.0;
    block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    // Step 3: requantize each sub-block scale to a signed 6-bit level against
    // the super-scale, re-derive that sub-block's 32 codebook levels against
    // the ROUNDED sub-scale, and pack both.
    let mut scales_h: u16 = 0;
    let mut scales_l = [0u8; NUM_SUB_BLOCKS / 2];
    let qs = &mut block[8..8 + SUPER_BLOCK_SIZE / 2];

    for ib in 0..NUM_SUB_BLOCKS {
        let l = nearest_int(id * scales[ib]).clamp(-32, 31);
        let dl = d * l as f32;
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };

        let sub = &xb[ib * SUB_BLOCK_SIZE..][..SUB_BLOCK_SIZE];
        let mut levels = [0usize; SUB_BLOCK_SIZE];
        for (j, &v) in sub.iter().enumerate() {
            levels[j] = best_index_int8(values, idl * v);
        }

        let biased = (l + 32) as u8;
        let l_l = biased & 0x0F;
        let l_h = biased >> 4;
        if ib % 2 == 0 {
            scales_l[ib / 2] = l_l;
        } else {
            scales_l[ib / 2] |= l_l << 4;
        }
        scales_h |= (l_h as u16) << (2 * ib);

        let sub_qs = &mut qs[ib * 16..(ib + 1) * 16];
        for j in 0..16 {
            sub_qs[j] = (levels[j] as u8) | ((levels[j + 16] as u8) << 4);
        }
    }

    block[2..4].copy_from_slice(&scales_h.to_le_bytes());
    block[4..8].copy_from_slice(&scales_l);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant_iq4::dequant_iq4_xs;

    #[test]
    fn block_size_is_136_bytes_covering_256_elements() {
        assert_eq!(BLOCK_BYTES, 136);
        assert_eq!(SUPER_BLOCK_SIZE, 256);
    }

    /// An all-zero super-block: every sub-block scale fits to `0.0`
    /// (`fit_subblock_scale`'s `GROUP_MAX_EPS` guard), so `max_scale == 0.0`
    /// and the super-scale `d = -0.0/32 == -0.0`. Every 6-bit level rounds to
    /// `0`, biased to `32` (`0x20`), so `scales_l` is `0x00` repeated and
    /// `scales_h` is `0b10` repeated 8 times = `0xAAAA`. `dl = d*0 = 0`, so
    /// every nibble lands on codebook index 8 (`0x88` repeated) — the same
    /// non-obvious degenerate pattern IQ4_NL's writer documents, not `0x00`.
    #[test]
    fn all_zero_super_block_matches_the_reference_degenerate_encoding() {
        let x = [0.0f32; SUPER_BLOCK_SIZE];
        let mut out = [0xFFu8; BLOCK_BYTES];
        quantize_iq4_xs(&x, &mut out);

        assert_eq!(
            f16::from_le_bytes([out[0], out[1]]).to_bits(),
            f16::from_f32(-0.0f32).to_bits()
        );
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), 0xAAAA);
        assert_eq!(&out[4..8], &[0u8; 4]);
        assert!(out[8..136].iter().all(|&b| b == 0x88));

        let mut back = [1.0f32; SUPER_BLOCK_SIZE];
        dequant_iq4_xs(&out, &mut back);
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
        let x = ramp(SUPER_BLOCK_SIZE);
        let mut out = [0u8; BLOCK_BYTES];
        quantize_iq4_xs(&x, &mut out);

        let mut back = [0.0f32; SUPER_BLOCK_SIZE];
        dequant_iq4_xs(&out, &mut back);

        for (got, want) in back.iter().zip(&x) {
            assert!((got - want).abs() < 0.6, "got {got}, want {want}");
        }
    }

    /// Every packed 6-bit sub-scale is recoverable from `scales_l`/`scales_h`
    /// — pack then unpack all 8 and compare, using the reader's own
    /// assembly (`sl | (sh << 4)`) rather than re-deriving it here.
    #[test]
    fn every_packed_six_bit_subscale_round_trips() {
        let x = ramp(SUPER_BLOCK_SIZE);
        let mut out = [0u8; BLOCK_BYTES];
        quantize_iq4_xs(&x, &mut out);

        let scales_h = u16::from_le_bytes([out[2], out[3]]);
        let scales_l = &out[4..8];

        // Recompute the same 8 fitted+requantized levels this writer chose,
        // by mirroring `quantize_super_block`'s steps 1-3 exactly.
        let mut weight = [0.0f32; SUB_BLOCK_SIZE];
        let mut scales = [0.0f32; NUM_SUB_BLOCKS];
        for ib in 0..NUM_SUB_BLOCKS {
            let sub = &x[ib * SUB_BLOCK_SIZE..][..SUB_BLOCK_SIZE];
            for (j, &v) in sub.iter().enumerate() {
                weight[j] = v * v;
            }
            scales[ib] = fit_subblock_scale(sub, &weight, &KVALUES_IQ4NL, NTRY);
        }
        let mut max_scale = 0.0f32;
        let mut amax_scale = 0.0f32;
        for &s in &scales {
            let a = s.abs();
            if a > amax_scale {
                amax_scale = a;
                max_scale = s;
            }
        }
        let d = -max_scale / 32.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for ib in 0..NUM_SUB_BLOCKS {
            let want_l = nearest_int(id * scales[ib]).clamp(-32, 31) + 32;
            let sl = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let sh = (scales_h >> (2 * ib)) & 0x03;
            let got_l = i32::from(sl) | (i32::from(sh) << 4);
            assert_eq!(
                got_l, want_l,
                "sub-block {ib} 6-bit scale did not round-trip"
            );
        }
    }

    /// A spread input reaches many codebook levels across the super-block —
    /// guards against a writer collapsing every sub-block onto one level.
    #[test]
    fn a_spread_input_reaches_many_codebook_entries() {
        let x = ramp(SUPER_BLOCK_SIZE);
        let mut out = [0u8; BLOCK_BYTES];
        quantize_iq4_xs(&x, &mut out);

        let mut seen = [false; 16];
        for &byte in &out[8..136] {
            seen[usize::from(byte & 0x0F)] = true;
            seen[usize::from(byte >> 4)] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        assert!(
            distinct > 8,
            "a 256-value ramp reached only {distinct} levels"
        );
    }

    #[test]
    fn importance_weighted_path_runs_and_differs_from_unweighted() {
        // A skewed importance vector makes the weighted fit favor different
        // elements than plain x^2 weighting, on a super-block where the two
        // objectives disagree about which outlier to protect.
        let mut x = [0.3f32; SUPER_BLOCK_SIZE];
        x[0] = 6.0;
        x[1] = -5.5;
        let mut imatrix = [1.0f32; SUPER_BLOCK_SIZE];
        imatrix[1] = 50.0;

        let mut plain = [0u8; BLOCK_BYTES];
        quantize_iq4_xs(&x, &mut plain);

        let mut weighted = [0u8; BLOCK_BYTES];
        quantize_iq4_xs_imatrix(&x, &mut weighted, &imatrix);

        assert_ne!(plain, weighted);
    }
}
