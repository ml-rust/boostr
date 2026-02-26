pub mod dequant;
pub mod fused_int4_qkv;
pub mod fused_int4_swiglu;
pub mod int4_gemm;
pub mod nf4;
pub mod quant_matmul;

/// Common WGSL helper functions for quantized operations.
///
/// WGSL has no u8 array type — must use array<u32> and extract bytes via bitwise ops.
/// WGSL has no native f16 — need software f16_to_f32 via IEEE 754 bit manipulation.
///
/// NOTE: WGSL does not allow passing storage pointers to functions, so read_u8/read_i8/read_f16
/// are provided as Rust functions that generate inline WGSL expressions. Use them in format! strings.
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
            if (sign == 1u) { return bitcast<f32>(0xFF800000u); } // -inf
            return bitcast<f32>(0x7F800000u); // inf
        }
        return bitcast<f32>(0x7FC00000u); // NaN
    }

    // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    let f_exp = f32(i32(exp) - 15);
    let f_mant = 1.0 + f32(mant) / 1024.0;
    let val = f_mant * exp2(f_exp);
    if (sign == 1u) { return -val; }
    return val;
}
"#
}

/// Generate inline WGSL expression to read a u8 from a u32 storage buffer.
/// `buf_name` is the WGSL variable name for the storage buffer (array<u32>).
/// `byte_idx_expr` is a WGSL expression for the byte index.
/// Returns a WGSL expression of type u32.
pub fn read_u8_inline(buf_name: &str, byte_idx_expr: &str) -> String {
    format!(
        "(({buf}[{idx} / 4u] >> (({idx} % 4u) * 8u)) & 0xFFu)",
        buf = buf_name,
        idx = byte_idx_expr
    )
}

/// Generate inline WGSL expression to read an i8 from a u32 storage buffer.
pub fn read_i8_inline(buf_name: &str, byte_idx_expr: &str) -> String {
    let u8_expr = read_u8_inline(buf_name, byte_idx_expr);
    format!("select(i32({u8_expr}), i32({u8_expr}) - 256, {u8_expr} >= 128u)")
}

/// Generate inline WGSL expression to read an f16 from a u32 storage buffer.
pub fn read_f16_inline(buf_name: &str, byte_idx_expr: &str) -> String {
    format!(
        "f16_to_f32(({buf}[{idx} / 4u] >> (({idx} % 4u) * 8u)) & 0xFFFFu)",
        buf = buf_name,
        idx = byte_idx_expr
    )
}
