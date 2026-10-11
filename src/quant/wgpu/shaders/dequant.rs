//! WGSL shader generators for dequantization operations

use super::common_helpers;
use super::k_quants::{q4k_block, q6k_block};

/// Generate WGSL shader for Q4_0 dequantization.
///
/// Q4_0 block layout (18 bytes per block of 32 elements):
///   - bytes [0..2): f16 scale (delta)
///   - bytes [2..18): 16 bytes of packed 4-bit quantized values (2 per byte)
///
/// Each value: dequant = (nibble - 8) * delta
pub fn generate_dequant_q4_0_shader() -> String {
    format!(
        r#"// Dequantize Q4_0 blocks to f32
{helpers}

const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 18u;

struct DequantParams {{
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams;

@compute @workgroup_size(256)
fn dequant_q4_0(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block_idx = gid.x;
    if (block_idx >= params.num_blocks) {{
        return;
    }}

    let base_byte = block_idx * BLOCK_BYTES;
    let delta = f16_to_f32((input[base_byte / 4u] >> ((base_byte % 4u) * 8u)) & 0xFFFFu);

    let out_base = block_idx * BLOCK_SIZE;

    for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
        let u8_byte_idx = base_byte + 2u + i;
        let byte_val = ((input[u8_byte_idx / 4u] >> ((u8_byte_idx % 4u) * 8u)) & 0xFFu);
        let lo = byte_val & 0xFu;
        let hi = (byte_val >> 4u) & 0xFu;

        output[out_base + i] = (f32(lo) - 8.0) * delta;
        output[out_base + i + 16u] = (f32(hi) - 8.0) * delta;
    }}
}}
"#,
        helpers = common_helpers(),
    )
}

/// Generate WGSL shader for Q8_0 dequantization.
///
/// Q8_0 block layout (34 bytes per block of 32 elements):
///   - bytes [0..2): f16 scale (delta)
///   - bytes [2..34): 32 bytes of signed 8-bit quantized values
///
/// Each value: dequant = qs[i] * delta
pub fn generate_dequant_q8_0_shader() -> String {
    format!(
        r#"// Dequantize Q8_0 blocks to f32
{helpers}

const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

struct DequantParams {{
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams;

@compute @workgroup_size(256)
fn dequant_q8_0(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block_idx = gid.x;
    if (block_idx >= params.num_blocks) {{
        return;
    }}

    let base_byte = block_idx * BLOCK_BYTES;
    let delta = f16_to_f32((input[base_byte / 4u] >> ((base_byte % 4u) * 8u)) & 0xFFFFu);

    let out_base = block_idx * BLOCK_SIZE;

    for (var i: u32 = 0u; i < 32u; i = i + 1u) {{
        let i8_byte_idx = base_byte + 2u + i;
        let i8_u8 = ((input[i8_byte_idx / 4u] >> ((i8_byte_idx % 4u) * 8u)) & 0xFFu);
        let qs = select(i32(i8_u8), i32(i8_u8) - 256, i8_u8 >= 128u);
        output[out_base + i] = f32(qs) * delta;
    }}
}}
"#,
        helpers = common_helpers(),
    )
}

/// Generate WGSL shader for Q4_K dequantization.
///
/// Q4_K super-block layout (144 bytes per block of 256 elements):
///   - bytes [0..2): f16 d (super-block scale)
///   - bytes [2..4): f16 dmin (super-block minimum)
///   - bytes [4..16): 8 sub-block scales and mins, packed 6-bit
///   - bytes [16..144): 128 bytes of 4-bit values
///
/// One invocation decodes one block. See [`super::k_quants::q4k_block`].
pub fn generate_dequant_q4_k_shader() -> String {
    k_quant_dequant_shader("dequant_q4_k", 144, &q4k_block("input", "base", store))
}

/// Generate WGSL shader for Q6_K dequantization.
///
/// Q6_K super-block layout (210 bytes per block of 256 elements):
///   - bytes [0..128): low 4 bits (ql)
///   - bytes [128..192): high 2 bits (qh)
///   - bytes [192..208): 16 int8 scales
///   - bytes [208..210): f16 d (super-block scale)
///
/// One invocation decodes one block. See [`super::k_quants::q6k_block`].
pub fn generate_dequant_q6_k_shader() -> String {
    k_quant_dequant_shader("dequant_q6_k", 210, &q6k_block("input", "base", store))
}

/// The statement that writes element `idx` of the current block.
fn store(idx: &str, val: &str) -> String {
    format!("output[out_base + {idx}] = {val};")
}

/// A one-block-per-invocation dequant shader around `decode`.
fn k_quant_dequant_shader(entry: &str, block_bytes: u32, decode: &str) -> String {
    format!(
        r#"// Dequantize 256-element K-quant blocks to f32
{helpers}

const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = {block_bytes}u;

struct DequantParams {{
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams;

@compute @workgroup_size(256)
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block_idx = gid.x;
    if (block_idx >= params.num_blocks) {{
        return;
    }}
    let base = block_idx * BLOCK_BYTES;
    let out_base = block_idx * BLOCK_SIZE;
    {decode}
}}
"#,
        helpers = common_helpers(),
    )
}
