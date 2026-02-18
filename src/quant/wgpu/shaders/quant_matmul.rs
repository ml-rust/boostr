//! WGSL shader generators for quantized matmul operations
//!
//! Each thread computes one element of the output matrix by dequantizing
//! the weight row on-the-fly and computing the dot product with the activation row.

use super::common_helpers;

/// Generate WGSL shader for Q4_0 quantized matmul.
///
/// activation [M, K] × weight_q4_0 [N, K] → output [M, N]
/// Each thread handles one (m, n) output element.
pub fn generate_quant_matmul_q4_0_shader() -> String {
    format!(
        r#"// Quantized matmul: activation [M,K] x Q4_0 weight [N,K] -> output [M,N]
{helpers}

const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 18u;

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> activation: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(16, 16)
fn quant_matmul_q4_0(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = gid.x;
    let m = gid.y;

    if (m >= params.M || n >= params.N) {{
        return;
    }}

    let num_blocks_per_row = params.K / BLOCK_SIZE;
    let weight_row_base = n * num_blocks_per_row * BLOCK_BYTES;
    let act_row_base = m * params.K;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < num_blocks_per_row; b = b + 1u) {{
        let block_base = weight_row_base + b * BLOCK_BYTES;
        let delta = read_f16(&weight, block_base);
        let k_base = b * BLOCK_SIZE;

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            let byte_val = read_u8(&weight, block_base + 2u + i);
            let lo = byte_val & 0xFu;
            let hi = (byte_val >> 4u) & 0xFu;

            let w0 = (f32(lo) - 8.0) * delta;
            let w1 = (f32(hi) - 8.0) * delta;

            acc = acc + activation[act_row_base + k_base + i] * w0;
            acc = acc + activation[act_row_base + k_base + i + 16u] * w1;
        }}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        helpers = common_helpers(),
    )
}

/// Generate WGSL shader for Q8_0 quantized matmul.
pub fn generate_quant_matmul_q8_0_shader() -> String {
    format!(
        r#"// Quantized matmul: activation [M,K] x Q8_0 weight [N,K] -> output [M,N]
{helpers}

const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> activation: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(16, 16)
fn quant_matmul_q8_0(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = gid.x;
    let m = gid.y;

    if (m >= params.M || n >= params.N) {{
        return;
    }}

    let num_blocks_per_row = params.K / BLOCK_SIZE;
    let weight_row_base = n * num_blocks_per_row * BLOCK_BYTES;
    let act_row_base = m * params.K;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < num_blocks_per_row; b = b + 1u) {{
        let block_base = weight_row_base + b * BLOCK_BYTES;
        let delta = read_f16(&weight, block_base);
        let k_base = b * BLOCK_SIZE;

        for (var i: u32 = 0u; i < 32u; i = i + 1u) {{
            let qs = read_i8(&weight, block_base + 2u + i);
            let w = f32(qs) * delta;
            acc = acc + activation[act_row_base + k_base + i] * w;
        }}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        helpers = common_helpers(),
    )
}

/// Generate WGSL shader for Q4_K quantized matmul.
pub fn generate_quant_matmul_q4_k_shader() -> String {
    format!(
        r#"// Quantized matmul: activation [M,K] x Q4_K weight [N,K] -> output [M,N]
{helpers}

const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 144u;

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> activation: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(16, 16)
fn quant_matmul_q4_k(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = gid.x;
    let m = gid.y;

    if (m >= params.M || n >= params.N) {{
        return;
    }}

    let num_blocks_per_row = params.K / BLOCK_SIZE;
    let weight_row_base = n * num_blocks_per_row * BLOCK_BYTES;
    let act_row_base = m * params.K;

    var acc: f32 = 0.0;

    for (var blk: u32 = 0u; blk < num_blocks_per_row; blk = blk + 1u) {{
        let base = weight_row_base + blk * BLOCK_BYTES;
        let d = read_f16(&weight, base);
        let dmin = read_f16(&weight, base + 2u);
        let scales_base = base + 4u;
        let qs_base = base + 16u;
        let k_base = blk * BLOCK_SIZE;

        for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {{
            var sc: f32;
            var mn: f32;

            if (sb < 4u) {{
                let scale_byte = read_u8(&weight, scales_base + sb);
                sc = d * f32(scale_byte & 63u);
                let min_byte = read_u8(&weight, scales_base + sb + 4u);
                mn = dmin * f32(min_byte & 63u);
            }} else {{
                let sb_off = sb - 4u;
                let scale_byte_lo = read_u8(&weight, scales_base + sb);
                let scale_byte_hi = read_u8(&weight, scales_base + sb + 4u);
                sc = d * f32((scale_byte_lo & 63u) | ((scale_byte_hi & 0x03u) << 6u));
                mn = dmin * f32(((scale_byte_lo >> 6u) & 3u) | ((scale_byte_hi >> 2u) << 2u));
            }}

            let sub_qs = qs_base + sb * 16u;
            let sub_k = k_base + sb * 32u;

            for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
                let byte_val = read_u8(&weight, sub_qs + i);
                let lo = byte_val & 0xFu;
                let hi = (byte_val >> 4u) & 0xFu;

                let w0 = f32(lo) * sc - mn;
                let w1 = f32(hi) * sc - mn;

                acc = acc + activation[act_row_base + sub_k + i] * w0;
                acc = acc + activation[act_row_base + sub_k + i + 16u] * w1;
            }}
        }}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        helpers = common_helpers(),
    )
}

/// Generate WGSL shader for Q6_K quantized matmul.
pub fn generate_quant_matmul_q6_k_shader() -> String {
    format!(
        r#"// Quantized matmul: activation [M,K] x Q6_K weight [N,K] -> output [M,N]
{helpers}

const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> activation: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

@compute @workgroup_size(16, 16)
fn quant_matmul_q6_k(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = gid.x;
    let m = gid.y;

    if (m >= params.M || n >= params.N) {{
        return;
    }}

    let num_blocks_per_row = params.K / BLOCK_SIZE;
    let weight_row_base = n * num_blocks_per_row * BLOCK_BYTES;
    let act_row_base = m * params.K;

    var acc: f32 = 0.0;

    for (var blk: u32 = 0u; blk < num_blocks_per_row; blk = blk + 1u) {{
        let base = weight_row_base + blk * BLOCK_BYTES;
        let ql_base = base;
        let qh_base = base + 128u;
        let sc_base = base + 192u;
        let d = read_f16(&weight, base + 208u);
        let k_base = blk * BLOCK_SIZE;

        for (var sb: u32 = 0u; sb < 16u; sb = sb + 1u) {{
            let scale = read_i8(&weight, sc_base + sb);

            for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
                let elem_idx = sb * 16u + i;

                let ql_byte_idx = elem_idx / 2u;
                let ql_byte = read_u8(&weight, ql_base + ql_byte_idx);
                var ql_val: u32;
                if (elem_idx % 2u == 0u) {{
                    ql_val = ql_byte & 0xFu;
                }} else {{
                    ql_val = (ql_byte >> 4u) & 0xFu;
                }}

                let qh_byte_idx = elem_idx / 4u;
                let qh_byte = read_u8(&weight, qh_base + qh_byte_idx);
                let qh_shift = (elem_idx % 4u) * 2u;
                let qh_val = (qh_byte >> qh_shift) & 0x3u;

                let q = ql_val | (qh_val << 4u);
                let w = d * f32(scale) * (f32(q) - 32.0);

                acc = acc + activation[act_row_base + k_base + elem_idx] * w;
            }}
        }}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        helpers = common_helpers(),
    )
}
