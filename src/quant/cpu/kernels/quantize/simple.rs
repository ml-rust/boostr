//! Quantization writers for the simple 32-element block formats
//!
//! Q4_0, Q4_1, Q8_0. These have no sub-block structure, so there is nothing for
//! a scale search to search over: Q8_0 already measures 0.54% relative RMS
//! against a 0.57% theoretical floor. They stay a direct absmax (Q4_0, Q8_0) or
//! min/max (Q4_1) fit, matching `ggml-quants.c`.
//!
//! # Nibble ordering — the trap
//!
//! Element `j` and element `j + 16` share ONE byte: low nibble is the first
//! half of the block, high nibble the second half. They are NOT adjacent output
//! positions. Packing `out[2i]`/`out[2i+1]` instead permutes every weight
//! within the block while keeping shape, block count and tensor RMS intact — it
//! measured 140% error in compressr and failed nothing until a model produced
//! garbage. `dequant_simple.rs` documents the same split for the read side and
//! is the authority here.

use super::search::nearest_int;
use half::f16;

/// Q4_0: 32 elements, 18 bytes — 2-byte f16 `d`, then 16 nibble-pair bytes
///
/// Inverse of [`dequant_q4_0`](crate::quant::cpu::kernels::dequant_simple::dequant_q4_0),
/// which reads `d` at byte 0 and `qs` at bytes 2..18 with
/// `out[j] = (qs[j] & 0xF) - 8`, `out[j + 16] = (qs[j] >> 4) - 8`.
///
/// `d = max / -8`, where `max` is the SIGNED value of largest magnitude. The
/// negative divisor is what makes level 0 (the most negative) reachable.
pub fn quantize_q4_0(x: &[f32], out: &mut [u8]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }

        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());

        for j in 0..16 {
            // +8.5 then truncate is llama.cpp's round-and-bias in one step; the
            // upper clamp is 15, the lower one is implicit in the +8 bias.
            let lo = ((xb[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let hi = ((xb[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
            block[2 + j] = lo | (hi << 4);
        }
    }
}

/// Q4_1: 32 elements, 20 bytes — f16 `d`, f16 `m`, then 16 nibble-pair bytes
///
/// Inverse of [`dequant_q4_1`](crate::quant::cpu::kernels::dequant_simple::dequant_q4_1),
/// which reads `d` at byte 0, `m` at byte 2, `qs` at bytes 4..20 and computes
/// `out = d·q + m`. Note the sign: Q4_1's min is ADDED, unlike Q4_K's `dmin`
/// which is subtracted.
pub fn quantize_q4_1(x: &[f32], out: &mut [u8]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in xb {
            min = min.min(v);
            max = max.max(v);
        }

        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(min).to_le_bytes());

        for j in 0..16 {
            let lo = (((xb[j] - min) * id + 0.5) as i32).clamp(0, 15) as u8;
            let hi = (((xb[j + 16] - min) * id + 0.5) as i32).clamp(0, 15) as u8;
            block[4 + j] = lo | (hi << 4);
        }
    }
}

/// Q8_0: 32 elements, 34 bytes — 2-byte f16 `d`, then 32 signed bytes
///
/// Inverse of [`dequant_q8_0`](crate::quant::cpu::kernels::dequant_simple::dequant_q8_0),
/// which reads `d` at byte 0 and `qs` at bytes 2..34 with `out[i] = qs[i]·d`.
/// Unlike the 4-bit formats there is no split-half ordering: element `i` is
/// byte `i`.
pub fn quantize_q8_0(x: &[f32], out: &mut [u8]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = x.len() / BLOCK_SIZE;
    debug_assert_eq!(out.len(), num_blocks * BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * BLOCK_SIZE..][..BLOCK_SIZE];
        let block = &mut out[b * BLOCK_BYTES..][..BLOCK_BYTES];

        let amax = xb.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());

        for (j, &v) in xb.iter().enumerate() {
            block[2 + j] = nearest_int(v * id).clamp(-127, 127) as i8 as u8;
        }
    }
}
