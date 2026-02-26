//! WebGPU shaders for NF4 operations

/// NF4 dequantization shader
/// Bindings: 0=nf4_data(u32 backing u8), 1=absmax(f32), 2=output(f32), 3=params(uniform)
pub fn generate_nf4_dequant_shader() -> String {
    format!(
        r#"// NF4 dequantization: nf4_data [bytes] -> output [f32]

const NF4_CODEBOOK = array<f32, 16>(
    0.0, -1.0, -0.6961928, -0.5250730,
    -0.3949739, -0.2844144, -0.1848489, -0.0911179,
    0.0796013, 0.1609302, 0.2461123, 0.3379120,
    0.4407173, 0.5626170, 0.7229568, 1.0
);

struct Nf4DequantParams {{
    num_bytes: u32,
    blocksize: u32,
    _pad0: u32,
    _pad1: u32,
}}

@group(0) @binding(0) var<storage, read_write> nf4_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> absmax: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Nf4DequantParams;

@compute @workgroup_size(256)
fn nf4_dequant(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let byte_idx = gid.x;
    if (byte_idx >= params.num_bytes) {{ return; }}

    let byte_val = ((nf4_data[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu);
    let idx_lo = byte_val & 0xFu;
    let idx_hi = (byte_val >> 4u) & 0xFu;

    let elem_lo = byte_idx * 2u;
    let elem_hi = byte_idx * 2u + 1u;

    output[elem_lo] = NF4_CODEBOOK[idx_lo] * absmax[elem_lo / params.blocksize];
    output[elem_hi] = NF4_CODEBOOK[idx_hi] * absmax[elem_hi / params.blocksize];
}}
"#
    )
}

/// NF4 fused GEMM shader
/// Bindings: 0=input(f32), 1=nf4_weight(u32 backing u8), 2=absmax(f32), 3=output(f32), 4=params(uniform)
pub fn generate_nf4_gemm_shader() -> String {
    format!(
        r#"// NF4 fused GEMM: input [M,K] x dequant(nf4_weight [K,N]) -> output [M,N]

const NF4_CODEBOOK_G = array<f32, 16>(
    0.0, -1.0, -0.6961928, -0.5250730,
    -0.3949739, -0.2844144, -0.1848489, -0.0911179,
    0.0796013, 0.1609302, 0.2461123, 0.3379120,
    0.4407173, 0.5626170, 0.7229568, 1.0
);

struct Nf4GemmParams {{
    m: u32,
    k: u32,
    n: u32,
    blocksize: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> nf4_weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> absmax: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Nf4GemmParams;

@compute @workgroup_size(16, 16, 1)
fn nf4_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) {{ return; }}

    let k_packed = params.k / 2u;
    let weight_row_start = col * k_packed;
    let absmax_row_start = col * (params.k / params.blocksize);

    var acc: f32 = 0.0;
    for (var bi: u32 = 0u; bi < k_packed; bi = bi + 1u) {{
        let nf4w_byte_idx = weight_row_start + bi;
        let byte_val = ((nf4_weight[nf4w_byte_idx / 4u] >> ((nf4w_byte_idx % 4u) * 8u)) & 0xFFu);
        let idx_lo = byte_val & 0xFu;
        let idx_hi = (byte_val >> 4u) & 0xFu;

        let elem_lo = bi * 2u;
        let elem_hi = bi * 2u + 1u;

        let w_lo = NF4_CODEBOOK_G[idx_lo] * absmax[absmax_row_start + elem_lo / params.blocksize];
        let w_hi = NF4_CODEBOOK_G[idx_hi] * absmax[absmax_row_start + elem_hi / params.blocksize];

        acc = acc + input[row * params.k + elem_lo] * w_lo + input[row * params.k + elem_hi] * w_hi;
    }}
    output[row * params.n + col] = acc;
}}
"#
    )
}
