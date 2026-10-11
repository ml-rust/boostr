//! WGSL block decoders for Q4_K and Q6_K, shared by dequant and matmul.
//!
//! Each generator emits the loops that decode one 256-element super-block
//! at a byte offset. `body(idx, val)` returns the WGSL statement that
//! consumes element `idx` (0..256, a `u32` expression) with value `val`
//! (an `f32` expression). The element order matches llama.cpp's
//! `dequantize_row_q4_K` and `dequantize_row_q6_K`, and boostr's CPU
//! kernels in `src/quant/cpu/kernels/dequant_k_quants/`.
//!
//! WGSL cannot pass a storage pointer to a function, so the decoders are
//! generated inline for a named `array<u32>` buffer.

use super::helpers::{read_f16_inline, read_i8_inline, read_u8_inline};

/// Q4_K: `d` f16, `dmin` f16, 12 packed 6-bit scale/min bytes, 128 bytes
/// of nibbles. Sub-block `j` (32 elements) reads the low nibbles of byte
/// run `j / 2` when `j` is even and the high nibbles when it is odd.
///
/// `base` is the WGSL byte offset of the block in `buf`.
pub fn q4k_block(buf: &str, base: &str, body: impl Fn(&str, &str) -> String) -> String {
    let byte = |offset: &str| read_u8_inline(buf, &format!("({base} + {offset})"));
    let s = |index: &str| byte(&format!("4u + {index}"));
    format!(
        r#"{{
    let q4k_d = {d};
    let q4k_dmin = {dmin};
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {{
        // llama.cpp `get_scale_min_k4`.
        var sc: u32;
        var mn: u32;
        if (j < 4u) {{
            sc = {s_j} & 63u;
            mn = {s_j4} & 63u;
        }} else {{
            sc = ({s_j4} & 0xFu) | (({s_jm4} >> 6u) << 4u);
            mn = ({s_j4} >> 4u) | (({s_j} >> 6u) << 4u);
        }}
        let dl = q4k_d * f32(sc);
        let ml = q4k_dmin * f32(mn);
        let run = 16u + (j / 2u) * 32u;
        let shift = (j % 2u) * 4u;
        for (var l: u32 = 0u; l < 32u; l = l + 1u) {{
            let q = ({q} >> shift) & 0xFu;
            {stmt}
        }}
    }}
}}"#,
        d = read_f16_inline(buf, base),
        dmin = read_f16_inline(buf, &format!("({base} + 2u)")),
        s_j = s("j"),
        s_j4 = s("(j + 4u)"),
        s_jm4 = s("(j - 4u)"),
        q = byte("run + l"),
        stmt = body("(j * 32u + l)", "(dl * f32(q) - ml)"),
    )
}

/// Q6_K: 128 bytes `ql`, 64 bytes `qh`, 16 i8 scales, `d` f16. Each half
/// (128 elements) uses 64 `ql` bytes, 32 `qh` bytes and 8 scales.
///
/// `base` is the WGSL byte offset of the block in `buf`.
pub fn q6k_block(buf: &str, base: &str, body: impl Fn(&str, &str) -> String) -> String {
    let byte = |offset: &str| read_u8_inline(buf, &format!("({base} + {offset})"));
    let scale = |offset: &str| read_i8_inline(buf, &format!("({base} + 192u + {offset})"));
    format!(
        r#"{{
    let q6k_d = {d};
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {{
        let ql = h * 64u;
        let qh = 128u + h * 32u;
        let sc = h * 8u;
        let out = h * 128u;
        for (var l: u32 = 0u; l < 32u; l = l + 1u) {{
            let is = l / 16u;
            let lo_a = {ql_a};
            let lo_b = {ql_b};
            let hi = {qh_l};
            let q1 = i32((lo_a & 0xFu) | ((hi & 3u) << 4u)) - 32;
            let q2 = i32((lo_b & 0xFu) | (((hi >> 2u) & 3u) << 4u)) - 32;
            let q3 = i32((lo_a >> 4u) | (((hi >> 4u) & 3u) << 4u)) - 32;
            let q4 = i32((lo_b >> 4u) | (((hi >> 6u) & 3u) << 4u)) - 32;
            {s1}
            {s2}
            {s3}
            {s4}
        }}
    }}
}}"#,
        d = read_f16_inline(buf, &format!("({base} + 208u)")),
        ql_a = byte("ql + l"),
        ql_b = byte("ql + l + 32u"),
        qh_l = byte("qh + l"),
        s1 = body(
            "(out + l)",
            &format!("(q6k_d * f32({}) * f32(q1))", scale("sc + is"))
        ),
        s2 = body(
            "(out + l + 32u)",
            &format!("(q6k_d * f32({}) * f32(q2))", scale("sc + is + 2u"))
        ),
        s3 = body(
            "(out + l + 64u)",
            &format!("(q6k_d * f32({}) * f32(q3))", scale("sc + is + 4u"))
        ),
        s4 = body(
            "(out + l + 96u)",
            &format!("(q6k_d * f32({}) * f32(q4))", scale("sc + is + 6u"))
        ),
    )
}
