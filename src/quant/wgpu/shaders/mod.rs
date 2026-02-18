pub mod dequant;
pub mod quant_matmul;

/// Common WGSL helper functions for quantized operations.
///
/// WGSL has no u8 array type — must use array<u32> and extract bytes via bitwise ops.
/// WGSL has no native f16 — need software f16_to_f32 via IEEE 754 bit manipulation.
pub fn common_helpers() -> &'static str {
    r#"
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;

    if (exp == 0u) {
        if (mant == 0u) {
            // Zero
            if (sign == 1u) { return -0.0; }
            return 0.0;
        }
        // Subnormal: value = (-1)^sign * 2^-14 * (mant / 1024)
        let val = f32(mant) / 1024.0 * 0.00006103515625; // 2^-14
        if (sign == 1u) { return -val; }
        return val;
    }
    if (exp == 31u) {
        // Inf or NaN
        if (mant == 0u) {
            if (sign == 1u) { return -1.0 / 0.0; } // -inf
            return 1.0 / 0.0; // inf
        }
        return 0.0 / 0.0; // NaN
    }

    // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    let f_exp = f32(i32(exp) - 15);
    let f_mant = 1.0 + f32(mant) / 1024.0;
    let val = f_mant * exp2(f_exp);
    if (sign == 1u) { return -val; }
    return val;
}

fn read_u8(data: ptr<storage, array<u32>, read_write>, byte_idx: u32) -> u32 {
    let word = (*data)[byte_idx / 4u];
    return (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

fn read_i8(data: ptr<storage, array<u32>, read_write>, byte_idx: u32) -> i32 {
    let val = read_u8(data, byte_idx);
    if (val >= 128u) { return i32(val) - 256; }
    return i32(val);
}

fn read_f16(data: ptr<storage, array<u32>, read_write>, byte_idx: u32) -> f32 {
    // f16 is 2 bytes, must be 2-byte aligned within the u32 word
    let word = (*data)[byte_idx / 4u];
    let shift = (byte_idx % 4u) * 8u;
    let bits = (word >> shift) & 0xFFFFu;
    return f16_to_f32(bits);
}
"#
}
