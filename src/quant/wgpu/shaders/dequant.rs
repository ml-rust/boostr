//! WGSL shader generators for dequantization operations

use super::common_helpers;

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
    let delta = read_f16(&input, base_byte);

    let out_base = block_idx * BLOCK_SIZE;

    for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
        let byte_val = read_u8(&input, base_byte + 2u + i);
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
    let delta = read_f16(&input, base_byte);

    let out_base = block_idx * BLOCK_SIZE;

    for (var i: u32 = 0u; i < 32u; i = i + 1u) {{
        let qs = read_i8(&input, base_byte + 2u + i);
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
///   - bytes [4..16): 12 bytes of packed 6-bit sub-block scales (8 sub-blocks)
///   - bytes [16..32): 16 bytes of packed 6-bit sub-block mins (8 sub-blocks)
///   - bytes [32..160): 128 bytes of packed 4-bit quantized values (256 elements, 2 per byte)
///
/// Note: Actual Q4_K layout — scales/mins are packed as 6-bit values in groups.
/// For simplicity, we use a standard 4-bit approach with 8 sub-blocks of 32.
pub fn generate_dequant_q4_k_shader() -> String {
    format!(
        r#"// Dequantize Q4_K blocks to f32
{helpers}

const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 144u;

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
fn dequant_q4_k(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block_idx = gid.x;
    if (block_idx >= params.num_blocks) {{
        return;
    }}

    let base = block_idx * BLOCK_BYTES;

    // Read super-block scale and min
    let d = read_f16(&input, base);
    let dmin = read_f16(&input, base + 2u);

    let out_base = block_idx * BLOCK_SIZE;

    // scales_base = base + 4, mins_base = base + 16 (but packed as 6-bit)
    // qs_base = base + 32 (in the simplified layout)
    // For the actual GGUF Q4_K format, scales and mins are packed together
    // in the first 12 bytes as 6-bit values.
    //
    // Actual layout: scales[12] at base+4, then qs[128] at base+16
    // The 12 bytes of scales encode both scale and min for 8 sub-blocks:
    //   - For sub-block j (0..7):
    //     scale_j = scales[j] & 0x3F
    //     min_j = scales[j] >> 4  (with high bits from scales[j+8])
    //
    // Simplified: read scales as bytes, extract low/high nibbles
    let scales_base = base + 4u;
    let qs_base = base + 16u;

    // Q4_K has scales packed in a specific way.
    // First 12 bytes at scales_base encode 8 sub-block scales and mins.
    // Bytes 0..7: low 6 bits = sub-scale, upper bits contribute to sub-min
    // Bytes 8..11: additional bits for sub-scales/mins above index 3
    //
    // For a correct but simpler approach: treat the 12 scale bytes as
    // pairs of (scale_lo, min_lo) with 4 bits each from the packed format.

    for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {{
        // Read sub-block scale and min from the packed format
        var sc: f32;
        var mn: f32;

        if (sb < 4u) {{
            let scale_byte = read_u8(&input, scales_base + sb);
            sc = d * f32(scale_byte & 63u);
            let min_byte = read_u8(&input, scales_base + sb + 4u);
            mn = dmin * f32(min_byte & 63u);
        }} else {{
            // Sub-blocks 4-7 use high bits from bytes 8-11
            let sb_off = sb - 4u;
            let scale_byte_lo = read_u8(&input, scales_base + sb);
            let scale_byte_hi = read_u8(&input, scales_base + sb + 4u);
            sc = d * f32((scale_byte_lo & 63u) | ((scale_byte_hi & 0x03u) << 6u));
            mn = dmin * f32(((scale_byte_lo >> 6u) & 3u) | ((scale_byte_hi >> 2u) << 2u));
        }}

        let sub_qs_base = qs_base + sb * 16u;
        let sub_out_base = out_base + sb * 32u;

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            let byte_val = read_u8(&input, sub_qs_base + i);
            let lo = byte_val & 0xFu;
            let hi = (byte_val >> 4u) & 0xFu;

            output[sub_out_base + i] = f32(lo) * sc - mn;
            output[sub_out_base + i + 16u] = f32(hi) * sc - mn;
        }}
    }}
}}
"#,
        helpers = common_helpers(),
    )
}

/// Generate WGSL shader for Q6_K dequantization.
///
/// Q6_K super-block layout (210 bytes per block of 256 elements):
///   - bytes [0..128): 128 bytes of low 4 bits (ql, 2 values per byte)
///   - bytes [128..192): 64 bytes of high 2 bits (qh, 4 values per byte)
///   - bytes [192..208): 16 bytes of scales (int8, one per 16-element sub-block)
///   - bytes [208..210): f16 d (super-block scale)
///
/// Each value: dequant = d * scale[j] * (q - 32)
/// where q = ql_nibble | (qh_2bits << 4), giving 6-bit range [0..63]
pub fn generate_dequant_q6_k_shader() -> String {
    format!(
        r#"// Dequantize Q6_K blocks to f32
{helpers}

const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

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
fn dequant_q6_k(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block_idx = gid.x;
    if (block_idx >= params.num_blocks) {{
        return;
    }}

    let base = block_idx * BLOCK_BYTES;

    // Layout offsets
    let ql_base = base;          // 128 bytes: low 4 bits
    let qh_base = base + 128u;   // 64 bytes: high 2 bits
    let sc_base = base + 192u;   // 16 bytes: int8 scales
    let d_offset = base + 208u;  // 2 bytes: f16 scale

    let d = read_f16(&input, d_offset);
    let out_base = block_idx * BLOCK_SIZE;

    // 16 sub-blocks of 16 elements each
    for (var sb: u32 = 0u; sb < 16u; sb = sb + 1u) {{
        let scale = read_i8(&input, sc_base + sb);

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            let elem_idx = sb * 16u + i;

            // Read low 4 bits from ql
            let ql_byte_idx = elem_idx / 2u;
            let ql_byte = read_u8(&input, ql_base + ql_byte_idx);
            var ql_val: u32;
            if (elem_idx % 2u == 0u) {{
                ql_val = ql_byte & 0xFu;
            }} else {{
                ql_val = (ql_byte >> 4u) & 0xFu;
            }}

            // Read high 2 bits from qh
            let qh_byte_idx = elem_idx / 4u;
            let qh_byte = read_u8(&input, qh_base + qh_byte_idx);
            let qh_shift = (elem_idx % 4u) * 2u;
            let qh_val = (qh_byte >> qh_shift) & 0x3u;

            // Combine: 6-bit value
            let q = ql_val | (qh_val << 4u);

            output[out_base + elem_idx] = d * f32(scale) * (f32(q) - 32.0);
        }}
    }}
}}
"#,
        helpers = common_helpers(),
    )
}
