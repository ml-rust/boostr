//! Quantization codebook tables and lookup functions.
//!
//! Provides shared codebook constants and quantize/dequantize functions
//! used by both boostr's internal kernels and downstream crates (compressr).

use super::cpu::kernels::nf4::NF4_CODEBOOK;

// ── IQ Codebook Tables ──────────────────────────────────────────────

/// Non-linear 4-bit codebook for IQ4_NL format.
/// Maps 4-bit indices (0..15) to quantized values.
/// These values are used as scale multipliers in the dequantization process.
/// Source: llama.cpp ggml-quants.c kvalues_iq4nl
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Quantize a normalized f32 value to a 4-bit NF4 index.
///
/// Input should be normalized to [-1, 1] range (divide by block absmax first).
/// Returns a 4-bit index (0..15) into the NF4 codebook.
pub fn quantize_nf4(x: f32) -> u8 {
    // Binary search for the nearest codebook entry
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;

    for (i, &val) in NF4_CODEBOOK.iter().enumerate() {
        let dist = (x - val).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }

    best_idx
}

/// Dequantize a 4-bit NF4 index to a normalized f32 value.
///
/// Returns the codebook value for the given index.
/// Multiply by block absmax to get the original scale.
pub fn dequantize_nf4(idx: u8) -> f32 {
    NF4_CODEBOOK[(idx & 0x0F) as usize]
}

/// Convert f32 to FP8 E4M3 format.
///
/// E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits.
/// Range: [-448, 448], bias = 7.
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0 {
        return (sign << 7) as u8;
    }
    if exp == 0xFF {
        return ((sign << 7) | 0x7E) as u8;
    }

    let new_exp = exp - 127 + 7;

    if new_exp <= 0 {
        return (sign << 7) as u8;
    }
    if new_exp >= 15 {
        return ((sign << 7) | 0x7E) as u8;
    }

    let new_mant = (mant >> 20) & 0x7;

    ((sign << 7) | ((new_exp as u32) << 3) | new_mant) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_round_trip() {
        for i in 0..16u8 {
            let val = dequantize_nf4(i);
            let back = quantize_nf4(val);
            assert_eq!(back, i, "NF4 round-trip failed for index {i}: val={val}");
        }
    }

    #[test]
    fn test_nf4_nearest() {
        // 0.0 should map to index 0
        assert_eq!(quantize_nf4(0.0), 0);
        // 1.0 should map to index 15
        assert_eq!(quantize_nf4(1.0), 15);
        // -1.0 should map to index 1
        assert_eq!(quantize_nf4(-1.0), 1);
    }

    #[test]
    fn test_fp8_e4m3_basic() {
        assert_eq!(f32_to_fp8_e4m3(0.0), 0);
        // Positive value
        let fp8 = f32_to_fp8_e4m3(1.0);
        assert_ne!(fp8, 0);
        assert_eq!(fp8 & 0x80, 0); // positive sign
        // Negative value
        let fp8_neg = f32_to_fp8_e4m3(-1.0);
        assert_ne!(fp8_neg & 0x80, 0); // negative sign
    }
}
