//! Q6_K quantization writer
//!
//! 256-element super-block, 16 sub-blocks of 16. Each sub-block gets a signed
//! scale from the symmetric search in [`super::search::make_qx_quants`]; the 16
//! scales are then quantized to 8 bits against one f16 super-block `d`. There
//! is no min — Q6_K reconstructs `x ≈ d · sc_j · (q − 32)`.
//!
//! # Field order — the trap
//!
//! Q6_K is the one GGML block whose scale does NOT come first. The layout is
//! `ql`@0..128, `qh`@128..192, 16 signed `scales`@192..208, f16 `d`@208..210.
//! Writing GGML's fields in declaration order from memory produced a file where
//! every dequantized value came back NaN.
//! [`dequant_q6k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q6k)
//! is the authority for these offsets.

#[cfg(test)]
use super::search::make_qx_absmax;
use super::search::{GROUP_MAX_EPS, make_qx_quants, nearest_int};
use half::f16;

const SUPER_BLOCK: usize = 256;
const BLOCK_BYTES: usize = 210;
const SUB_BLOCKS: usize = 16;
/// Q6_K levels span `[-32, 31]` and are stored biased by `+32`
const NMAX: i32 = 32;

/// Per-sub-block scale fit: `(values, nmax, biased_levels) -> scale`
type ScaleFit = fn(&[f32], i32, &mut [u8]) -> f32;

/// Q6_K: 256 elements, 210 bytes
///
/// Inverse of [`dequant_q6k`](crate::quant::cpu::kernels::dequant_k_quants::dequant_q6k).
pub fn quantize_q6k(x: &[f32], out: &mut [u8]) {
    quantize_q6k_with(x, out, make_qx_quants)
}

/// Q6_K with an explicit scale fit — lets tests measure the search against absmax
pub(super) fn quantize_q6k_with(x: &[f32], out: &mut [u8], fit: ScaleFit) {
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
        for ib in 0..SUB_BLOCKS {
            let scale = fit(&xb[16 * ib..][..16], NMAX, &mut levels[16 * ib..][..16]);
            scales[ib] = scale;
            if scale.abs() > max_abs_scale {
                max_abs_scale = scale.abs();
                max_scale = scale;
            }
        }
        // An all-zero super-block stays all zero: d = 0 makes every level moot.
        if max_abs_scale < GROUP_MAX_EPS {
            continue;
        }

        // Scales are SIGNED 8-bit. The negative divisor mirrors the symmetric
        // level range, so the sub-block carrying `max_scale` lands on -128.
        let iscale = -128.0 / max_scale;
        let d = f16::from_f32(1.0 / iscale);
        block[208..210].copy_from_slice(&d.to_le_bytes());
        for ib in 0..SUB_BLOCKS {
            block[192 + ib] = nearest_int(iscale * scales[ib]).clamp(-128, 127) as i8 as u8;
        }

        // Second pass: levels against the scale the READER reconstructs, i.e.
        // the f16 `d` times the 8-bit scale, not the exact float from the fit.
        for ib in 0..SUB_BLOCKS {
            let dl = d.to_f32() * (block[192 + ib] as i8) as f32;
            if dl == 0.0 {
                continue;
            }
            for j in 0..16 {
                let l = nearest_int(xb[16 * ib + j] / dl).clamp(-NMAX, NMAX - 1);
                levels[16 * ib + j] = (l + NMAX) as u8;
            }
        }

        pack_q6k(&levels, block);
    }
}

/// Split the 6-bit levels into 4 low bits in `ql` and 2 high bits in `qh`
///
/// Per 128-element half `n`, for `l` in `0..32` the reader takes:
/// `ql[64n + l]` low nibble + `qh[32n + l]` bits 0-1 → element `128n + l`;
/// `ql[64n + l + 32]` low + bits 2-3 → element `128n + l + 32`;
/// `ql[64n + l]` high nibble + bits 4-5 → element `128n + l + 64`;
/// `ql[64n + l + 32]` high + bits 6-7 → element `128n + l + 96`.
fn pack_q6k(levels: &[u8; SUPER_BLOCK], block: &mut [u8]) {
    for n in 0..2 {
        let base = n * 128;
        let ql = n * 64;
        let qh = 128 + n * 32;
        for l in 0..32 {
            let q1 = levels[base + l];
            let q2 = levels[base + l + 32];
            let q3 = levels[base + l + 64];
            let q4 = levels[base + l + 96];
            block[ql + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
            block[ql + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
            block[qh + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
}

/// Absmax baseline for Q6_K — the scale choice the search replaces.
///
/// Exists only so the tests can assert the search actually beats it; nothing
/// in the shipped path should ever pick the worse scale on purpose.
#[cfg(test)]
pub(super) fn quantize_q6k_absmax(x: &[f32], out: &mut [u8]) {
    quantize_q6k_with(x, out, make_qx_absmax)
}
