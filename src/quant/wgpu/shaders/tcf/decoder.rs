//! The WGSL decoder body: every read-direction rule and every reconstruction
//! expression the TCF kernels share.
//!
//! Generated once by [`decoder`] and pasted into both entry points, so the two
//! kernels cannot drift on a bit position or on a rounding. The text depends on
//! a module-scope `payload: array<u32>` binding and a `params: TcfParams`
//! uniform, both declared by `super::kernels`.

use tcf_core::{MAX_SUB_MIN_6, MAX_SUB_SCALE, MAX_SUB_SCALE_6};

use crate::quant::tcf::{
    SCALE_FORM_FLAT, SCALE_FORM_TWO_LEVEL_U6M6, SCALE_FORM_TWO_LEVEL_U8, TCF_TILE,
};

/// Every read-direction rule and every reconstruction expression, as WGSL.
///
/// Depends on a module-scope `payload: array<u32>` binding and a
/// `params: TcfParams` uniform: WGSL cannot take a storage pointer as a
/// function parameter, so the decoder names its buffer directly.
pub(super) fn decoder() -> String {
    format!(
        r#"
const TCF_TILE: u32 = {tile}u;
const TCF_SUPER_BLOCK_TILES: u32 = 4u;
const TCF_SCALE_FLAT: u32 = {flat}u;
const TCF_SCALE_TWO_LEVEL_U8: u32 = {two_u8}u;
const TCF_SCALE_TWO_LEVEL_U6M6: u32 = {two_u6m6}u;
const TCF_SUB_SCALE_LEVELS_U8: u32 = {levels_u8}u;
const TCF_SUB_SCALE_LEVELS_U6: u32 = {levels_u6}u;
const TCF_SUB_MIN_LEVELS_U6: u32 = {min_levels_u6}u;
const TCF_SUB_MIN_SIGN_6: i32 = {min_levels_u6};
const TCF_SUB_MIN_SPAN_6: i32 = 64;

// One byte of the payload. WGSL has no u8, so a byte index is split into a
// word index and a shift.
fn tcf_byte(index: u32) -> u32 {{
    return (payload[index >> 2u] >> ((index & 3u) * 8u)) & 0xFFu;
}}

// A little-endian binary16 taken apart into an EXACT integer significand and
// a power of two: the value's magnitude is `mant * 2^exp2`.
//
// A binary16 is an integer significand times a power of two in every class but
// infinity and NaN, so this loses nothing. Both readers below start here, so
// the field extraction is written once.
struct TcfBin16 {{
    // Integer significand, with the implicit bit restored for a normal.
    // At most 11 bits, so `mant * sub` stays exact in f32 and in u32.
    mant: u32,
    // Power of two the significand carries.
    exp2: i32,
    // `0x80000000` for a negative value, `0` otherwise.
    sign: u32,
    // `0u` for an infinity or a NaN, `1u` otherwise.
    finite: u32,
}}

fn tcf_binary16_parts(index: u32) -> TcfBin16 {{
    let bits = tcf_byte(index) | (tcf_byte(index + 1u) << 8u);
    let exp16 = (bits >> 10u) & 0x1Fu;
    let mant16 = bits & 0x3FFu;

    var out: TcfBin16;
    out.sign = (bits & 0x8000u) << 16u;
    out.finite = select(1u, 0u, exp16 == 0x1Fu);
    if (exp16 == 0u || exp16 == 0x1Fu) {{
        // Zero, subnormal, or a non-finite payload: no implicit bit. A
        // subnormal is `mant16 * 2^-24`.
        out.mant = mant16;
        out.exp2 = -24;
    }} else {{
        // Normal: restore the implicit bit. Value is `mant * 2^(exp16 - 25)`,
        // binary16 bias 15 less the 10 stored fraction bits.
        out.mant = mant16 | 0x400u;
        out.exp2 = i32(exp16) - 25;
    }}
    return out;
}}

// A little-endian binary16 at an arbitrary byte offset, decoded as f32.
//
// Pure integer bit manipulation, mirroring `tcf_core::binary16::bits_to_f32`:
// infinity and NaN keep their payload, and zero, subnormal and normal are one
// normalize-and-encode path over the exact significand. Every binary16 is
// exactly representable in f32, and no float operation appears here, so the
// result cannot differ from the CPU value by a rounding mode or by an
// implementation's `exp2` accuracy.
fn tcf_binary16(index: u32) -> f32 {{
    let parts = tcf_binary16_parts(index);
    if (parts.finite == 0u) {{
        // Infinity when the payload is zero, NaN otherwise; the payload is
        // widened into the f32 mantissa's high bits, as the reference does.
        return bitcast<f32>(parts.sign | 0x7F800000u | (parts.mant << 13u));
    }}
    if (parts.mant == 0u) {{
        return bitcast<f32>(parts.sign);
    }}
    // `k` is the index of the significand's highest set bit, 0..=10. Shifting
    // it to bit 23 normalizes a subnormal and leaves a normal alone.
    let k = 31u - countLeadingZeros(parts.mant);
    let frac32 = (parts.mant << (23u - k)) & 0x7FFFFFu;
    let biased = i32(k) + parts.exp2 + 127;
    return bitcast<f32>(parts.sign | (u32(biased) << 23u) | frac32);
}}

// `(binary16 at `index` * signed_factor) / den`, CORRECTLY ROUNDED, computed
// entirely in integers.
//
// # Why this is not a float divide
//
// The CPU reference is `(bits_to_f32(super) * f32(sub)) / den` in
// `QuantLayout::group_scale` and `QuantLayout::group_min`, an IEEE f32 divide
// and so correctly rounded. WGSL specifies `x / y` at 2.5 ULP, NOT correctly
// rounded, and a real adapter was measured one ULP off on `Q6S16D_T64`.
//
// WGSL's `fma` cannot repair it. Section 15.7.4's accuracy table gives
// `fma(x, y, z)` the accuracy of `x * y + z`, so a single rounding is
// PERMITTED, never required, and naga's HLSL backend emits `mad` while its
// GLSL backend expands to `(x * y + z)` when the target lacks `fma`. A
// Markstein refinement built on it would be exact on some adapters and not on
// others, which is the bug being fixed.
//
// # Why integers are enough
//
// The numerator is exact and small. `super` is a binary16, so its significand
// is at most 11 bits, and the factor is at most an 8-bit magnitude, so their
// integer product `num` needs at most 19 bits. The quotient is then a rational
// with a small integer numerator and a constant denominator, and its correctly
// rounded f32 follows from restoring long division plus a round-to-nearest-even
// on the 25th significant bit — no float operation, so no adapter's division
// accuracy can enter.
//
// `magnitude` is the factor's absolute value and `negative` is `1u` when the
// factor is negative; rounding to nearest even is sign-symmetric, so the sign
// is carried through untouched.
fn tcf_scaled_quotient(index: u32, magnitude: u32, negative: u32, den: u32) -> f32 {{
    let parts = tcf_binary16_parts(index);
    var sign = parts.sign;
    if (negative != 0u) {{
        sign = sign ^ 0x80000000u;
    }}

    if (parts.finite == 0u) {{
        // An infinity or a NaN super-value, which no payload `tcf-core`
        // accepted carries. Keep the function total by falling back to the
        // float expression rather than encoding a wrong exponent.
        var factor = f32(magnitude);
        if (negative != 0u) {{
            factor = -factor;
        }}
        return (tcf_binary16(index) * factor) / f32(den);
    }}

    // Exact: at most 11 bits times at most 8 bits.
    let num = parts.mant * magnitude;
    if (num == 0u) {{
        // `(+-x) * 0` is a zero of the product's sign, and dividing it keeps
        // that sign. `tcf_value` adds this minimum, so the sign matters.
        return bitcast<f32>(sign);
    }}

    // Restoring long division of `num / den` down to 25 significant bits,
    // keeping the remainder as the sticky bit. `rem` stays below `2 * den`,
    // and `q` stays below `2^25`, so neither can overflow a u32.
    var rem = num % den;
    var q = num / den;
    var shift: i32 = 0;
    // `num >= 1` and `den <= 255`, so the quotient reaches 2^24 within 32
    // steps; the bound only makes termination syntactic.
    for (var iteration: i32 = 0; iteration < 64; iteration = iteration + 1) {{
        if (q >= 0x1000000u) {{
            break;
        }}
        rem = rem << 1u;
        q = q << 1u;
        if (rem >= den) {{
            rem = rem - den;
            q = q + 1u;
        }}
        shift = shift + 1;
    }}

    // `q` now holds 25 significant bits of `num / den * 2^shift`, and `rem`
    // is nonzero exactly when more nonzero bits follow. Round the 25th bit
    // away, to nearest with ties to even.
    let guard = q & 1u;
    var mant = q >> 1u;
    if (guard == 1u && (rem != 0u || (mant & 1u) == 1u)) {{
        mant = mant + 1u;
    }}
    // Magnitude is `mant * 2^pow2`, with `mant` in [2^23, 2^24] before the
    // carry check and in [2^23, 2^24) after it.
    var pow2 = parts.exp2 + 1 - shift;
    if (mant == 0x1000000u) {{
        mant = 0x800000u;
        pow2 = pow2 + 1;
    }}

    let biased = pow2 + 150;
    if (biased < 1 || biased > 254) {{
        // Outside the f32 normal range. Unreachable for a TCF payload: the
        // smallest nonzero magnitude here is 2^-24 / 255 and the largest is
        // 65504 * 255. Fall back rather than encode a wrong exponent.
        var factor = f32(magnitude);
        if (negative != 0u) {{
            factor = -factor;
        }}
        return (tcf_binary16(index) * factor) / f32(den);
    }}
    return bitcast<f32>(sign | (u32(biased) << 23u) | (mant - 0x800000u));
}}

// Section 14.6 read direction. Field `slot` of super-block `block` occupies
// bits [6*slot, 6*slot+6) of that block's byte run, LSB-first, and a field
// starting past bit 2 continues into the next byte.
fn tcf_packed6(plane_off: u32, block: u32, slot: u32) -> u32 {{
    let base = plane_off + block * params.sub_block_bytes;
    let bit = slot * 6u;
    let byte_index = base + (bit >> 3u);
    let offset = bit & 7u;
    var value = tcf_byte(byte_index) >> offset;
    if (offset > 2u) {{
        value = value | (tcf_byte(byte_index + 1u) << (8u - offset));
    }}
    return value & 0x3Fu;
}}

// One group's effective dequantization parameters. A symmetric group has no
// minimum and yields 0.0, which `tcf_value` must not add.
struct TcfGroup {{
    scale: f32,
    min_value: f32,
}}

// The effective (scale, minimum) of group `g` of tile `tile`.
//
// Section 13.0 / Section 13.0.1: a flat layout stores both outright as
// binary16. Section 13.3: a two-level u8 layout stores a u8 sub-scale under a
// per-super-block binary16 super-scale. Section 13.4: a two-level 6-bit layout
// stores a 6-bit unsigned sub-scale and a 6-bit signed sub-minimum, both
// bit-packed per super-block, under a super-scale and a super-minimum.
//
// Multiply then divide once, in that fixed order, matching
// `QuantLayout::group_scale` and `QuantLayout::group_min`. The product is
// exact, so the division is the expression's only rounding, and
// `tcf_scaled_quotient` does both in integers so that rounding is the CPU's.
fn tcf_group_values(tile: u32, g: u32) -> TcfGroup {{
    let block = tile / TCF_SUPER_BLOCK_TILES;
    let slot = (tile % TCF_SUPER_BLOCK_TILES) * params.groups_per_tile + g;
    let global = tile * params.groups_per_tile + g;
    var out: TcfGroup;

    if (params.scale_form == TCF_SCALE_TWO_LEVEL_U8) {{
        let sub = tcf_byte(params.scale_off + global);
        out.scale = tcf_scaled_quotient(
            params.super_off + block * 2u, sub, 0u, TCF_SUB_SCALE_LEVELS_U8);
        out.min_value = 0.0;
        return out;
    }}
    if (params.scale_form == TCF_SCALE_TWO_LEVEL_U6M6) {{
        let sub = tcf_packed6(params.scale_off, block, slot);
        out.scale = tcf_scaled_quotient(
            params.super_off + block * 2u, sub, 0u, TCF_SUB_SCALE_LEVELS_U6);

        let field = tcf_packed6(params.min_off, block, slot);
        // A 6-bit two's-complement field: 0..=31 is itself, 32..=63 is
        // `field - 64`. The reserved -32 never reaches here in a payload
        // `tcf-core` accepted.
        var level = i32(field);
        if (level > TCF_SUB_MIN_SIGN_6) {{
            level = level - TCF_SUB_MIN_SPAN_6;
        }}
        out.min_value = tcf_scaled_quotient(
            params.super_min_off + block * 2u,
            u32(abs(level)),
            select(0u, 1u, level < 0),
            TCF_SUB_MIN_LEVELS_U6);
        return out;
    }}

    out.scale = tcf_binary16(params.scale_off + global * 2u);
    if (params.symmetric != 0u) {{
        out.min_value = 0.0;
    }} else {{
        out.min_value = tcf_binary16(params.min_off + global * 2u);
    }}
    return out;
}}

// The code of element `e` of tile `tile`, already sign-resolved.
//
// Section 14.1 / Section 14.1.1: a 4-bit tile is a 32-byte run, element `e` in
// byte `e / 2`, low nibble when `e` is even. Section 14.3: an 8-bit tile is one
// byte per element. Section 14.2: a 6-bit code plane is a whole low-nibble
// sub-plane followed by a whole high-two-bit sub-plane, the second starting at
// `code_high_off`.
//
// Symmetric codes sign-extend from `bits`; Section 13.2's reserved
// most-negative pattern is rejected by `tcf-core` when the payload is read, so
// shader code sign-extends the legal range rather than re-checking per element.
fn tcf_code(tile: u32, e: u32) -> i32 {{
    var field: u32 = 0u;
    if (params.bits == 4u) {{
        let byte_value = tcf_byte(tile * (TCF_TILE / 2u) + (e >> 1u));
        if ((e & 1u) != 0u) {{
            field = (byte_value >> 4u) & 0x0Fu;
        }} else {{
            field = byte_value & 0x0Fu;
        }}
    }} else if (params.bits == 8u) {{
        field = tcf_byte(tile * TCF_TILE + e);
    }} else {{
        let low_byte = tcf_byte(tile * (TCF_TILE / 2u) + (e >> 1u));
        var low: u32 = low_byte & 0x0Fu;
        if ((e & 1u) != 0u) {{
            low = (low_byte >> 4u) & 0x0Fu;
        }}
        let high_byte = tcf_byte(params.code_high_off + tile * (TCF_TILE / 4u) + (e >> 2u));
        let top = (high_byte >> ((e & 3u) * 2u)) & 0x03u;
        field = low | (top << 4u);
    }}

    if (params.symmetric == 0u) {{
        return i32(field);
    }}
    let reserved = i32(1u << (params.bits - 1u));
    let value = i32(field);
    if (value > reserved) {{
        return value - 2 * reserved;
    }}
    return value;
}}

// `x`, forced to exist as a concrete, already-rounded f32 before whatever
// consumes it.
//
// Section 15.7.4 leaves `a * b + c` free to CONTRACT into a single rounding,
// and gives `fma(x, y, z)` the accuracy of `x * y + z`, so neither form is
// pinned by the specification. An adapter this project was measured on fused
// `d * code + m` for `Q4AS32D_T64` and returned `0x3f450002` where the CPU's
// two separate roundings give `0x3f450000` — two ULP of the sum, which is
// exactly half an ULP of the product `d * 14`, the signature of one fused
// rounding rather than two.
//
// A `let` binding does not forbid that. Only breaking the value's float
// dependency chain does, and `bitcast` is the one break WGSL offers: naga
// lowers it to a real instruction on every backend — `OpBitcast` on SPIR-V,
// `asuint`/`asfloat` on HLSL, `as_type` on MSL, `floatBitsToUint` on GLSL —
// and folds none of them, its constant evaluator rejecting bitcast outright.
//
// The `| params.zero_barrier` is what keeps a downstream LLVM-based driver
// from folding the two bitcasts back into nothing and contracting anyway. The
// host writes zero into that field on every dispatch, so the bit pattern is
// unchanged, and no optimizer can prove the value of a uniform.
fn tcf_settled(x: f32) -> f32 {{
    return bitcast<f32>(bitcast<u32>(x) | params.zero_barrier);
}}

// Section 13.0 / Section 13.0.1 applied to one element, against its group's
// already-resolved parameters.
//
// Symmetric and asymmetric are separate expressions rather than one with a
// zero minimum: `-0.0 + 0.0` is `+0.0`, so folding them would change the sign
// of a zero the CPU path emits. A symmetric group is one multiply and needs no
// barrier; the asymmetric product goes through `tcf_settled` so the multiply
// and the add round separately, as `d * code + m` does on the CPU.
fn tcf_value(code: i32, scale: f32, min_value: f32) -> f32 {{
    if (params.symmetric != 0u) {{
        return scale * f32(code);
    }}
    let product = tcf_settled(scale * f32(code));
    return product + min_value;
}}
"#,
        tile = TCF_TILE,
        flat = SCALE_FORM_FLAT,
        two_u8 = SCALE_FORM_TWO_LEVEL_U8,
        two_u6m6 = SCALE_FORM_TWO_LEVEL_U6M6,
        levels_u8 = MAX_SUB_SCALE,
        levels_u6 = MAX_SUB_SCALE_6,
        min_levels_u6 = MAX_SUB_MIN_6,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sub-scale and sub-minimum level counts the shader divides by are
    /// `tcf-core`'s, never restated numbers.
    #[test]
    fn the_level_counts_come_from_tcf_core() {
        let source = decoder();
        assert!(source.contains(&format!("TCF_SUB_SCALE_LEVELS_U8: u32 = {MAX_SUB_SCALE}u")));
        assert!(source.contains(&format!(
            "TCF_SUB_SCALE_LEVELS_U6: u32 = {MAX_SUB_SCALE_6}u"
        )));
        assert!(source.contains(&format!("TCF_SUB_MIN_LEVELS_U6: u32 = {MAX_SUB_MIN_6}u")));
        assert!(source.contains(&format!("TCF_SUB_MIN_SIGN_6: i32 = {MAX_SUB_MIN_6};")));
    }

    /// The scale-form discriminants the shader tests against are the host's,
    /// never restated numbers.
    #[test]
    fn the_scale_form_discriminants_come_from_the_host() {
        let source = decoder();
        assert!(source.contains(&format!("TCF_SCALE_FLAT: u32 = {SCALE_FORM_FLAT}u")));
        assert!(source.contains(&format!(
            "TCF_SCALE_TWO_LEVEL_U8: u32 = {SCALE_FORM_TWO_LEVEL_U8}u"
        )));
        assert!(source.contains(&format!(
            "TCF_SCALE_TWO_LEVEL_U6M6: u32 = {SCALE_FORM_TWO_LEVEL_U6M6}u"
        )));
    }

    /// The asymmetric reconstruction never spells `d * code + m` as one
    /// expression a backend is free to contract into a single rounding. The
    /// product goes through the bitcast barrier first.
    #[test]
    fn the_asymmetric_product_is_pinned_before_the_add() {
        let source = decoder();
        assert!(source.contains("let product = tcf_settled(scale * f32(code));"));
        assert!(source.contains("return product + min_value;"));
        assert!(!source.contains("scale * f32(code) + min_value"));
    }

    /// The barrier is a bitcast round trip ORed with the host's always-zero
    /// uniform field, which is what a driver that folds a bare bitcast pair
    /// cannot see through.
    #[test]
    fn the_barrier_ors_the_zero_uniform_into_the_bit_pattern() {
        let source = decoder();
        assert!(source.contains("bitcast<f32>(bitcast<u32>(x) | params.zero_barrier)"));
    }
}
