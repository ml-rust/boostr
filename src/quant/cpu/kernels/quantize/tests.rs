//! Round-trip and accuracy tests for the quantization writers
//!
//! Every test dequantizes with boostr's OWN reader kernel rather than a private
//! decoder. A writer paired with its own reader can agree with itself while
//! disagreeing with the format — that is exactly how three layout bugs (Q6_K
//! field order, Q4_0/Q4_1 nibble pairing) shipped elsewhere in this codebase.
//!
//! The input is several super-blocks long with genuinely varying magnitude
//! across sub-blocks. A constant or single-block input passes with a wrong
//! interleave, because a within-block permutation preserves shape, block count
//! and tensor RMS.

use super::q4k_q5k::{KSearch, Q4K_SEARCH, Q5K_SEARCH, quantize_q4k_with, quantize_q5k_with};
use super::q6k::quantize_q6k_absmax;
use super::{
    quantize_q4_0, quantize_q4_1, quantize_q4k, quantize_q5k, quantize_q6k, quantize_q8_0,
};
use crate::quant::cpu::kernels::dequant_k_quants::{dequant_q4k, dequant_q5k, dequant_q6k};
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
fn synthetic_weights() -> Vec<f32> {
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
fn relative_rms(reference: &[f32], decoded: &[f32]) -> f32 {
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
fn round_trip(
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

#[test]
fn q4k_round_trip() {
    let x = synthetic_weights();
    round_trip(&x, 144, 256, quantize_q4k, dequant_q4k, 0.072);
}

#[test]
fn q5k_round_trip() {
    let x = synthetic_weights();
    round_trip(&x, 176, 256, quantize_q5k, dequant_q5k, 0.04);
}

#[test]
fn q6k_round_trip() {
    let x = synthetic_weights();
    round_trip(&x, 210, 256, quantize_q6k, dequant_q6k, 0.02);
}

/// A single hand-built super-block pins the Q6_K field order.
///
/// Writing GGML's fields in declaration order (`d` first) instead of
/// `ql`/`qh`/`scales`/`d` makes every dequantized value NaN — the f16 read at
/// byte 208 lands in the middle of the level data. Reading a constant back
/// exactly is enough to catch it.
#[test]
fn q6k_field_order_matches_the_reader() {
    let x = vec![0.25f32; 256];
    let mut packed = vec![0u8; 210];
    quantize_q6k(&x, &mut packed);

    let mut decoded = vec![0.0f32; 256];
    dequant_q6k(&packed, &mut decoded);
    for (i, &v) in decoded.iter().enumerate() {
        assert!((v - 0.25).abs() < 0.01, "elem {i}: expected 0.25, got {v}");
    }
}

/// Beating absmax is the whole point of the unit, so it is asserted, not assumed.
///
/// The baseline is the same pipeline with the sweep disabled (`nstep = 0`),
/// which is exactly the plain min/max fit compressr shipped. Any regression in
/// the search shows up here before it shows up in a converted checkpoint.
#[test]
fn q4k_search_beats_absmax() {
    let x = synthetic_weights();
    let no_sweep = KSearch {
        nstep: 0,
        ..Q4K_SEARCH
    };

    let mut searched = vec![0u8; (x.len() / 256) * 144];
    let mut absmax = vec![0u8; searched.len()];
    quantize_q4k_with(&x, &mut searched, &Q4K_SEARCH);
    quantize_q4k_with(&x, &mut absmax, &no_sweep);

    let (a, b) = decode_pair(&x, &searched, &absmax, dequant_q4k);
    assert!(a < b, "q4_k: search {a} must beat absmax {b}");
}

#[test]
fn q5k_search_beats_absmax() {
    let x = synthetic_weights();
    let no_sweep = KSearch {
        nstep: 0,
        ..Q5K_SEARCH
    };

    let mut searched = vec![0u8; (x.len() / 256) * 176];
    let mut absmax = vec![0u8; searched.len()];
    quantize_q5k_with(&x, &mut searched, &Q5K_SEARCH);
    quantize_q5k_with(&x, &mut absmax, &no_sweep);

    let (a, b) = decode_pair(&x, &searched, &absmax, dequant_q5k);
    assert!(a < b, "q5_k: search {a} must beat absmax {b}");
}

/// Dequantize two packings of the same input, returning both relative RMS values
fn decode_pair(
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

/// Same claim for Q6_K, whose search is the symmetric routine instead.
#[test]
fn q6k_search_beats_absmax() {
    let x = synthetic_weights();
    let blocks = x.len() / 256;

    let mut searched = vec![0u8; blocks * 210];
    let mut absmax = vec![0u8; blocks * 210];
    quantize_q6k(&x, &mut searched);
    quantize_q6k_absmax(&x, &mut absmax);

    let mut ds = vec![0.0f32; x.len()];
    let mut da = vec![0.0f32; x.len()];
    dequant_q6k(&searched, &mut ds);
    dequant_q6k(&absmax, &mut da);

    let searched_rms = relative_rms(&x, &ds);
    let absmax_rms = relative_rms(&x, &da);
    assert!(
        searched_rms < absmax_rms,
        "q6_k: search {searched_rms} must beat absmax {absmax_rms}"
    );
}
