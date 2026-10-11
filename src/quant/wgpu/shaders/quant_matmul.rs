//! WGSL shader generators for quantized matmul operations
//!
//! Each thread computes one element of the output matrix by dequantizing
//! the weight row on-the-fly and computing the dot product with the activation row.

use super::common_helpers;
use super::k_quants::{q4k_block, q6k_block};

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
        let delta = f16_to_f32((weight[block_base / 4u] >> ((block_base % 4u) * 8u)) & 0xFFFFu);
        let k_base = b * BLOCK_SIZE;

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            let u8_byte_idx = block_base + 2u + i;
            let byte_val = ((weight[u8_byte_idx / 4u] >> ((u8_byte_idx % 4u) * 8u)) & 0xFFu);
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
        let delta = f16_to_f32((weight[block_base / 4u] >> ((block_base % 4u) * 8u)) & 0xFFFFu);
        let k_base = b * BLOCK_SIZE;

        for (var i: u32 = 0u; i < 32u; i = i + 1u) {{
            let i8_byte_idx = block_base + 2u + i;
            let i8_u8 = ((weight[i8_byte_idx / 4u] >> ((i8_byte_idx % 4u) * 8u)) & 0xFFu);
            let qs = select(i32(i8_u8), i32(i8_u8) - 256, i8_u8 >= 128u);
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
///
/// activation [M, K] × weight_q4_k [N, K] → output [M, N]. Each thread
/// handles one (m, n) output element. See [`super::k_quants::q4k_block`].
pub fn generate_quant_matmul_q4_k_shader() -> String {
    k_quant_matmul_shader(
        "quant_matmul_q4_k",
        144,
        &q4k_block("weight", "base", accumulate),
    )
}

/// Generate WGSL shader for Q6_K quantized matmul.
///
/// activation [M, K] × weight_q6_k [N, K] → output [M, N]. Each thread
/// handles one (m, n) output element. See [`super::k_quants::q6k_block`].
pub fn generate_quant_matmul_q6_k_shader() -> String {
    k_quant_matmul_shader(
        "quant_matmul_q6_k",
        210,
        &q6k_block("weight", "base", accumulate),
    )
}

/// The statement that adds element `idx` of the current block to `acc`.
fn accumulate(idx: &str, val: &str) -> String {
    format!("acc = acc + activation[act_row_base + k_base + {idx}] * {val};")
}

/// A one-output-per-thread matmul shader around `decode`.
fn k_quant_matmul_shader(entry: &str, block_bytes: u32, decode: &str) -> String {
    format!(
        r#"// Quantized matmul: activation [M,K] x K-quant weight [N,K] -> output [M,N]
{helpers}

const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = {block_bytes}u;

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
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
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
        let k_base = blk * BLOCK_SIZE;
        {decode}
    }}

    output[m * params.N + n] = acc;
}}
"#,
        helpers = common_helpers(),
    )
}
