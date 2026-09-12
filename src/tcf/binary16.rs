//! IEEE 754 binary16 (half-precision) conversion.
//!
//! TCF stores every quantization scale and minimum as a binary16 (Section 12.2,
//! Section 13, Section 14). Section 13 requires quantized codes to be computed against the
//! *stored* binary16 scale, never a higher-precision intermediate, so this
//! module is normative for every quantized tensor's byte content.
//!
//! Conversion is pure integer bit manipulation on `f32::to_bits` /
//! `f32::from_bits` output — never floating-point arithmetic on the value
//! itself — so rounding is fully controlled by this code, not the
//! compiler's own f32 arithmetic.

// f32 layout (IEEE 754 binary32).
const F32_SIGN_MASK: u32 = 0x8000_0000;
const F32_EXP_MASK: u32 = 0xFF;
const F32_MANT_MASK: u32 = 0x007F_FFFF;
const F32_IMPLICIT_BIT: u32 = 0x0080_0000;
const F32_EXP_BIAS: i32 = 127;

// binary16 layout (IEEE 754 binary16).
const F16_SIGN_MASK: u16 = 0x8000;
const F16_EXP_FIELD_MASK: u16 = 0x1F; // 5-bit exponent field width
const F16_EXP_INF_NAN: u32 = 0x1F; // all-ones exponent field: inf/NaN
const F16_MANT_MASK: u32 = 0x03FF;
const F16_EXP_INF_NAN_BITS: u16 = 0x7C00; // exponent field 0x1F, shifted into place
const F16_MAX_NORMAL_EXP: i32 = 30; // largest finite binary16 exponent field
const F16_MANT_BITS: u32 = 10; // stored fraction bits
const F16_IMPLICIT_MANT_OVERFLOW: u32 = 1 << (F16_MANT_BITS + 1); // 0x0800: carry into next exponent
const F16_SUBNORMAL_TO_NORMAL: u32 = 1 << F16_MANT_BITS; // 0x0400: smallest normal mantissa
const F16_EXP_BIAS: i32 = 15;
// Largest unbiased normal exponent (binary16 normal range is -14..=15).
// Numerically equal to `F16_EXP_BIAS`, but that equality is a property of
// the binary16 format, not a coincidence to fold away.
const F16_MAX_UNBIASED_EXP: i32 = 15;
// Smallest unbiased normal exponent (2^-14).
const F16_MIN_UNBIASED_EXP: i32 = -14;
// f32 mantissa (23 bits) minus binary16 mantissa (10 bits): bits dropped
// when narrowing a normal-range significand.
const NORMAL_DROP_BITS: u32 = 13;
// f32 exponent bias (127) minus binary16 exponent bias (15): rebias shift
// between binary32 and binary16 normal exponents.
const EXP_BIAS_DELTA: u32 = 112;
// f32 exponent field corresponding to binary16's subnormal range base
// (2^-14, encoded as f32 exponent 113, minus the leading-bit index it is
// paired with in `bits_to_f32`'s renormalization: 113 - 10 = 103).
const F16_SUBNORMAL_EXP32_BASE: u32 = 103;

/// Round an f32 to IEEE 754 binary16, returning the bit pattern (Section 12.2).
///
/// Section 13 requires this exact function (round-to-nearest, ties-to-even, in
/// both the normal and subnormal range) to be the one used to compute
/// quantization codes against the stored scale — never a higher-precision
/// intermediate.
///
/// Overflow (magnitude >= 65520, the midpoint between the largest finite
/// binary16 value 65504 and 65536) rounds to infinity. Underflow (magnitude
/// < 2^-25, the midpoint between zero and the smallest subnormal 2^-24)
/// rounds to zero, sign preserved. NaN inputs stay NaN.
#[must_use]
pub const fn f32_to_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits & F32_SIGN_MASK) >> 16) as u16;
    let exp32 = (bits >> 23) & F32_EXP_MASK;
    let mant32 = bits & F32_MANT_MASK;

    // Class 1: infinity or NaN (f32 exponent field all-ones).
    if exp32 == F32_EXP_MASK {
        if mant32 != 0 {
            // Preserve a non-zero payload so the result stays NaN rather
            // than collapsing to infinity.
            let mut payload = (mant32 >> NORMAL_DROP_BITS) as u16;
            if payload == 0 {
                payload = 1;
            }
            return sign | F16_EXP_INF_NAN_BITS | payload;
        }
        return sign | F16_EXP_INF_NAN_BITS;
    }

    // Class 2: zero or f32 subnormal (f32 exponent field zero). The largest
    // f32 subnormal is ~2^-126, far below the binary16 underflow threshold
    // of 2^-25, so this always flushes to signed zero.
    if exp32 == 0 {
        return sign;
    }

    // Normal f32: restore the implicit leading 1 into a 24-bit significand.
    let signif = mant32 | F32_IMPLICIT_BIT;
    let e = exp32 as i32 - F32_EXP_BIAS;

    if e > F16_MAX_UNBIASED_EXP {
        // Magnitude >= 2^16 = 65536, unconditionally overflows.
        return sign | F16_EXP_INF_NAN_BITS;
    }

    // Class 3: binary16 normal range (exponent field 1..=30).
    if e >= F16_MIN_UNBIASED_EXP {
        // Rounding a 24-bit significand down to 11 bits (implicit + 10
        // fraction). The 13 discarded bits split into a round bit and a
        // sticky bit.
        let mut he = e + F16_EXP_BIAS;
        let round_bit = (signif >> (NORMAL_DROP_BITS - 1)) & 1;
        let sticky = (signif & ((1 << (NORMAL_DROP_BITS - 1)) - 1)) != 0;
        let mut kept = signif >> NORMAL_DROP_BITS;
        if round_bit != 0 && (sticky || (kept & 1) != 0) {
            kept += 1;
        }
        if kept == F16_IMPLICIT_MANT_OVERFLOW {
            // Mantissa overflowed into the next exponent.
            kept >>= 1;
            he += 1;
        }
        if he > F16_MAX_NORMAL_EXP {
            // Rounded past the largest finite binary16 (e.g. 65520 ties to
            // this case, since its even neighbor is infinity, not 65504).
            return sign | F16_EXP_INF_NAN_BITS;
        }
        let mantissa16 = (kept & F16_MANT_MASK) as u16;
        return sign | ((he as u16) << F16_MANT_BITS) | mantissa16;
    }

    // Class 4: e <= -15, binary16 subnormal range, or underflow to zero.
    // Align the 24-bit significand to units of 2^-24 (the subnormal ULP).
    let rshift = (-e - 1) as u32; // >= 14
    if rshift >= 25 {
        // Round bit would come from beyond the significand's 24 bits, so
        // it is always 0: always rounds down to zero.
        return sign;
    }
    let kept0 = signif >> rshift;
    let round_bit_pos = rshift - 1;
    let rem_mask = (1u32 << rshift) - 1;
    let rem = signif & rem_mask;
    let round_bit = (rem >> round_bit_pos) & 1;
    let sticky_mask = (1u32 << round_bit_pos) - 1;
    let sticky = (rem & sticky_mask) != 0;

    let mut kept = kept0;
    if round_bit != 0 && (sticky || (kept & 1) != 0) {
        kept += 1;
    }
    if kept == F16_SUBNORMAL_TO_NORMAL {
        // Rounded up into the smallest normal value.
        return sign | (1u16 << F16_MANT_BITS);
    }
    sign | (kept as u16)
}

/// Expand an IEEE 754 binary16 bit pattern to f32, exactly (Section 12.2).
///
/// This is the inverse used to interpret a stored binary16 scale (Section 13) back
/// into an f32 for display or further (non-normative) computation; the
/// widening is lossless for every input, including NaN payloads.
#[must_use]
pub const fn bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & F16_SIGN_MASK) as u32) << 16;
    let exp16 = ((bits >> F16_MANT_BITS) & F16_EXP_FIELD_MASK) as u32;
    let mant16 = (bits as u32) & F16_MANT_MASK;

    // Class 1: infinity or NaN (binary16 exponent field all-ones).
    if exp16 == F16_EXP_INF_NAN {
        if mant16 == 0 {
            return f32::from_bits(sign | (F32_EXP_MASK << 23));
        }
        // NaN: widen the payload into the f32 mantissa's high bits.
        let mant32 = mant16 << NORMAL_DROP_BITS;
        return f32::from_bits(sign | (F32_EXP_MASK << 23) | mant32);
    }

    // Class 2: zero or binary16 subnormal (exponent field zero).
    if exp16 == 0 {
        if mant16 == 0 {
            return f32::from_bits(sign);
        }
        // Binary16 subnormal: normalize into an f32 normal number. `k` is
        // the bit index of the highest set bit in `mant16` (0..=9).
        let lz = mant16.leading_zeros();
        let k = 31 - lz;
        let frac_bits = mant16 & ((1u32 << k) - 1);
        let frac32 = frac_bits << (23 - k);
        let exp32 = k + F16_SUBNORMAL_EXP32_BASE;
        return f32::from_bits(sign | (exp32 << 23) | frac32);
    }

    // Class 3/4: binary16 normal: rebias the exponent (binary16 bias 15,
    // f32 bias 127) and widen the mantissa by zero-padding.
    let exp32 = exp16 + EXP_BIAS_DELTA;
    let frac32 = mant16 << NORMAL_DROP_BITS;
    f32::from_bits(sign | (exp32 << 23) | frac32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_nan_bits(bits: u16) -> bool {
        (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0
    }

    /// NaN payloads are not required to round-trip bit-for-bit (Section 12.2 only
    /// requires NaN-ness to survive), so this asserts NaN-in implies
    /// NaN-out on both sides of the conversion instead of bit equality.
    fn assert_nan_stays_nan(bits: u16) {
        assert!(
            bits_to_f32(bits).is_nan(),
            "bits {bits:#06x} should decode to NaN"
        );
        assert!(
            bits_to_f32(f32_to_bits(bits_to_f32(bits))).is_nan(),
            "NaN bits {bits:#06x} must stay NaN through a round trip"
        );
    }

    /// Every non-NaN binary16 bit pattern must survive `bits_to_f32` then
    /// `f32_to_bits` unchanged; every NaN bit pattern must stay NaN.
    /// Cross-validated against an external oracle: do not weaken this test
    /// to make a future edit pass.
    #[test]
    fn every_binary16_bit_pattern_round_trips_or_stays_nan() {
        for bits in 0u16..=0xFFFF {
            if is_nan_bits(bits) {
                assert_nan_stays_nan(bits);
                continue;
            }
            let value = bits_to_f32(bits);
            assert_eq!(
                f32_to_bits(value),
                bits,
                "round trip failed for bits {bits:#06x} (value {value})"
            );
        }
    }

    #[test]
    fn known_decimal_literals_match_spec_bit_patterns() {
        assert_eq!(f32_to_bits(1.0), 0x3c00);
        assert_eq!(f32_to_bits(0.5), 0x3800);
        assert_eq!(f32_to_bits(0.0), 0x0000);
        assert_eq!(f32_to_bits(-1.0), 0xbc00);
    }

    /// Builds an f32 that ties exactly between two binary16 values in the
    /// normal range: `kept0` is the 11-bit (implicit + 10 fraction)
    /// significand below the tie, and the exact midpoint sets the next
    /// lower bit (the round bit) with a zero sticky remainder.
    fn normal_range_tie(exp32: u32, kept0: u32) -> f32 {
        let signif = (kept0 << 13) | (1 << 12);
        let mant32 = signif & 0x007F_FFFF;
        f32::from_bits((exp32 << 23) | mant32)
    }

    #[test]
    fn ties_even_normal_range() {
        // e = 0 (values in [1, 2)): kept0 = 0x401 is odd, so the tie must
        // round up to the even neighbor 0x402.
        let value = normal_range_tie(127, 0x401);
        assert_eq!(f32_to_bits(value) & 1, 0, "tie must round to even mantissa");

        // e = 3 (values in [8, 16)): same odd-kept0 tie at a different
        // exponent.
        let value = normal_range_tie(130, 0x401);
        assert_eq!(f32_to_bits(value) & 1, 0, "tie must round to even mantissa");

        // A tie landing on an already-even kept0 must not move at all.
        let value = normal_range_tie(127, 0x400);
        assert_eq!(
            f32_to_bits(value),
            0x3C00,
            "tie on an even mantissa must not round up"
        );
    }

    #[test]
    fn subnormal_boundaries() {
        // Smallest normal: 2^-14.
        let smallest_normal = f32::from_bits(113u32 << 23);
        assert_eq!(f32_to_bits(smallest_normal), 0x0400);

        // Largest subnormal: 1023 * 2^-24.
        let largest_subnormal = bits_to_f32(0x03FF);
        assert_eq!(f32_to_bits(largest_subnormal), 0x03FF);

        // Smallest subnormal: 2^-24.
        let smallest_subnormal = f32::from_bits(103u32 << 23);
        assert_eq!(f32_to_bits(smallest_subnormal), 0x0001);

        // Exact midpoint at 2^-25: ties-even sends it to zero (0 is even).
        let midpoint = f32::from_bits(102u32 << 23);
        assert_eq!(f32_to_bits(midpoint), 0x0000);
        let midpoint_neg = f32::from_bits((102u32 << 23) | 0x8000_0000);
        assert_eq!(f32_to_bits(midpoint_neg), 0x8000);

        // Well below half the smallest subnormal: rounds to zero.
        let tiny = f32::from_bits(90u32 << 23);
        assert_eq!(f32_to_bits(tiny), 0x0000);
    }

    #[test]
    fn overflow_rounds_to_infinity() {
        assert_eq!(f32_to_bits(65504.0), 0x7BFF);
        assert_eq!(f32_to_bits(65520.0), 0x7C00);
        assert_eq!(f32_to_bits(65519.99), 0x7BFF);
    }

    #[test]
    fn signed_zero_and_underflow_sign() {
        assert_eq!(f32_to_bits(0.0), 0x0000);
        assert_eq!(f32_to_bits(-0.0), 0x8000);
        assert_eq!(f32_to_bits(-1e-9), 0x8000);
    }

    #[test]
    fn nan_stays_nan() {
        assert!(bits_to_f32(0x7C01).is_nan());
        assert!(bits_to_f32(0xFC01).is_nan());
        assert!(f32_to_bits(f32::NAN) & 0x7C00 == 0x7C00);
        assert!(f32_to_bits(f32::NAN) & 0x03FF != 0);
    }
}
