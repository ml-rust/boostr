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

/// Round-trip and accuracy fixtures for the quantization writers, shared by
/// every writer's tests in this module.
///
/// Every test dequantizes with boostr's OWN reader kernel rather than a private
/// decoder. A writer paired with its own reader can agree with itself while
/// disagreeing with the format — that is exactly how three layout bugs (Q6_K
/// field order, Q4_0/Q4_1 nibble pairing) shipped elsewhere in this codebase.
///
/// The input is several super-blocks long with genuinely varying magnitude
/// across sub-blocks. A constant or single-block input passes with a wrong
/// interleave, because a within-block permutation preserves shape, block count
/// and tensor RMS.
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::quant::cpu::kernels::dequant_simple::{dequant_q4_0, dequant_q4_1, dequant_q8_0};

    /// Number of 256-element super-blocks in the test input
    const SUPER_BLOCKS: usize = 8;
    const N: usize = SUPER_BLOCKS * 256;

    /// Seeded LCG (Numerical Recipes constants) — deterministic, no `rand` dep
    struct Lcg(u32);

    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Top 24 bits into [-1, 1).
            ((self.0 >> 8) as f32 / 8_388_608.0) - 1.0
        }
    }

    /// Weight-like input: roughly normal, with per-sub-block magnitude varying over
    /// three orders of magnitude and occasional outliers
    ///
    /// The magnitude spread is what makes a shared-scale bug visible: with uniform
    /// magnitudes a wrong sub-block-to-scale mapping costs almost nothing.
    pub(in super::super) fn synthetic_weights() -> Vec<f32> {
        let mut rng = Lcg(0x5EED_1234);
        let mut out = Vec::with_capacity(N);
        for i in 0..N {
            // Sum of three uniforms approximates a normal.
            let g = rng.next_unit() + rng.next_unit() + rng.next_unit();
            let sub = i / 32;
            let magnitude = 10.0f32.powi((sub % 7) as i32 - 3);
            let outlier = if i % 211 == 0 { 6.0 } else { 1.0 };
            out.push(g * magnitude * outlier);
        }
        out
    }

    /// Relative RMS of the reconstruction error: `‖x̂ − x‖ / ‖x‖`
    pub(in super::super) fn relative_rms(reference: &[f32], decoded: &[f32]) -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&r, &d) in reference.iter().zip(decoded) {
            num += ((d - r) as f64).powi(2);
            den += (r as f64).powi(2);
        }
        (num / den).sqrt() as f32
    }

    /// Quantize, dequantize with the matching reader, and check the shared
    /// invariants: exact packed size, all-finite output, error inside the band.
    pub(in super::super) fn round_trip(
        values: &[f32],
        block_bytes: usize,
        block_size: usize,
        quantize: impl Fn(&[f32], &mut [u8]),
        dequantize: fn(&[u8], &mut [f32]),
        max_rel_rms: f32,
    ) -> f32 {
        let num_blocks = values.len() / block_size;
        let mut packed = vec![0u8; num_blocks * block_bytes];
        quantize(values, &mut packed);
        assert_eq!(packed.len(), num_blocks * block_bytes, "packed size");

        let mut decoded = vec![0.0f32; values.len()];
        dequantize(&packed, &mut decoded);
        assert!(decoded.iter().all(|v| v.is_finite()), "non-finite output");

        let rms = relative_rms(values, &decoded);
        assert!(
            rms < max_rel_rms,
            "relative RMS {rms} exceeds the {max_rel_rms} band"
        );
        rms
    }

    /// Dequantize two packings of the same input, returning both relative RMS values
    pub(in super::super) fn decode_pair(
        x: &[f32],
        left: &[u8],
        right: &[u8],
        dequantize: fn(&[u8], &mut [f32]),
    ) -> (f32, f32) {
        let mut dl = vec![0.0f32; x.len()];
        let mut dr = vec![0.0f32; x.len()];
        dequantize(left, &mut dl);
        dequantize(right, &mut dr);
        (relative_rms(x, &dl), relative_rms(x, &dr))
    }

    #[test]
    fn q4_0_round_trip() {
        let x = synthetic_weights();
        round_trip(&x, 18, 32, quantize_q4_0, dequant_q4_0, 0.11);
    }

    #[test]
    fn q4_1_round_trip() {
        let x = synthetic_weights();
        round_trip(&x, 20, 32, quantize_q4_1, dequant_q4_1, 0.10);
    }

    #[test]
    fn q8_0_round_trip() {
        let x = synthetic_weights();
        round_trip(&x, 34, 32, quantize_q8_0, dequant_q8_0, 0.01);
    }

    /// The same claim for the single-scale formats, whose search is
    /// [`super::super::block_scale`].
    ///
    /// The baseline is the identical encoder with the sweep replaced by the plain
    /// absmax scale, so the sweep is the only difference between the two packings.
    /// A regression in the search shows up here before it shows up in a converted
    /// checkpoint.
    #[test]
    fn q4_0_search_beats_absmax() {
        let x = synthetic_weights();
        let mut searched = vec![0u8; (x.len() / 32) * 18];
        let mut absmax = vec![0u8; searched.len()];
        quantize_q4_0(&x, &mut searched);
        quantize_q4_0_absmax(&x, &mut absmax);

        let (a, b) = decode_pair(&x, &searched, &absmax, dequant_q4_0);
        assert!(a < b, "q4_0: search {a} must beat absmax {b}");
    }

    /// Q4_1's search is [`super::super::block_affine`], which sweeps the scale AND the
    /// added offset. Its baseline is the plain min/max fit through the identical
    /// encoder, so the sweep is the only difference between the two packings.
    #[test]
    fn q4_1_search_beats_minmax() {
        let x = synthetic_weights();
        let mut searched = vec![0u8; (x.len() / 32) * 20];
        let mut minmax = vec![0u8; searched.len()];
        quantize_q4_1(&x, &mut searched);
        quantize_q4_1_minmax(&x, &mut minmax);

        let (a, b) = decode_pair(&x, &searched, &minmax, dequant_q4_1);
        assert!(a < b, "q4_1: search {a} must beat min/max {b}");
    }

    /// The search moves the stored fields, never the format.
    ///
    /// Q4_1's min/max fit could only ever store a scale and offset drawn from the
    /// block itself. The search stores a LEAST-SQUARES pair instead, which can leave
    /// binary16's range on an extreme block. Both fields must still be values a
    /// reader can multiply and add, so both are checked on every block.
    #[test]
    fn q4_1_stored_fields_stay_finite() {
        let x = synthetic_weights();
        let mut packed = vec![0u8; (x.len() / 32) * 20];
        quantize_q4_1(&x, &mut packed);

        for (b, block) in packed.as_chunks::<20>().0.iter().enumerate() {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            assert!(
                d.is_finite() && m.is_finite(),
                "block {b} stores a non-finite field, which no reader can use"
            );
        }
    }

    #[test]
    fn q8_0_search_beats_absmax() {
        let x = synthetic_weights();
        let mut searched = vec![0u8; (x.len() / 32) * 34];
        let mut absmax = vec![0u8; searched.len()];
        quantize_q8_0(&x, &mut searched);
        quantize_q8_0_absmax(&x, &mut absmax);

        let (a, b) = decode_pair(&x, &searched, &absmax, dequant_q8_0);
        assert!(a < b, "q8_0: search {a} must beat absmax {b}");
    }

    /// The search moves the scale, never the format.
    ///
    /// Q8_0 codes are clamped to `[-127, 127]`, one short of the signed-byte range.
    /// A refit that widened the range to -128 would still decode, and would still
    /// round-trip inside the error band, but the block would no longer be
    /// symmetric and kernels that negate a code would overflow.
    #[test]
    fn q8_0_codes_stay_in_the_symmetric_range() {
        let x = synthetic_weights();
        let mut packed = vec![0u8; (x.len() / 32) * 34];
        quantize_q8_0(&x, &mut packed);

        for (b, block) in packed.as_chunks::<34>().0.iter().enumerate() {
            for (i, &byte) in block[2..].iter().enumerate() {
                let code = byte as i8;
                assert!(
                    code != -128,
                    "block {b} elem {i}: code -128 is out of range"
                );
            }
        }
    }

    /// A constant block must come back exactly, which pins Q4_0's sign convention.
    ///
    /// The scale takes the opposite sign to the largest-magnitude element, so a
    /// block of -0.75 gets a POSITIVE scale and lands on level -8. Getting that
    /// sign backwards still produces valid nibbles and a plausible tensor RMS, and
    /// only shows up as a reconstruction that misses by a factor near two.
    #[test]
    fn q4_0_round_trips_a_constant_block() {
        let x = vec![-0.75f32; 32];
        let mut packed = vec![0u8; 18];
        quantize_q4_0(&x, &mut packed);

        let mut decoded = vec![0.0f32; 32];
        dequant_q4_0(&packed, &mut decoded);
        for (i, &v) in decoded.iter().enumerate() {
            assert!((v + 0.75).abs() < 0.01, "elem {i}: expected -0.75, got {v}");
        }
    }
}
