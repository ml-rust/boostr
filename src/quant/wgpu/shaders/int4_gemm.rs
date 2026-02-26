//! WebGPU shaders for INT4 GEMM operations (AWQ, GPTQ, Marlin)

/// AWQ INT4 GEMM shader
/// input[M,K] x dequant(qweight[K,N/8]) -> output[M,N]
/// Bindings: 0=input(f32), 1=qweight(u32), 2=scales(f32), 3=zeros(f32), 4=output(f32), 5=params(uniform)
pub fn generate_int4_gemm_shader() -> String {
    format!(
        r#"// INT4 GEMM: input [M,K] x dequant(qweight [K,N/8]) -> output [M,N]

const AWQ_SHIFTS = array<u32, 8>(0u, 16u, 4u, 20u, 8u, 24u, 12u, 28u);

fn unpack_int4_awq(packed: u32, idx: u32) -> u32 {{
    return (packed >> AWQ_SHIFTS[idx]) & 0xFu;
}}

struct Int4GemmParams {{
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> qweight: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> zeros: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Int4GemmParams;

@compute @workgroup_size(16, 16, 1)
fn int4_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) {{ return; }}

    let n_packed = params.n / 8u;
    let pack_col = col / 8u;
    let sub = col % 8u;

    var acc: f32 = 0.0;
    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {{
        let a = input[row * params.k + ki];
        let packed = qweight[ki * n_packed + pack_col];
        let q = unpack_int4_awq(packed, sub);

        let group = ki / params.group_size;
        let scale = scales[group * params.n + col];
        let zero = zeros[group * params.n + col];
        let w = (f32(q) - zero) * scale;
        acc = acc + a * w;
    }}
    output[row * params.n + col] = acc;
}}
"#
    )
}

/// GPTQ INT4 GEMM shader
/// Bindings: 0=input(f32), 1=qweight(u32), 2=qzeros(u32), 3=scales(f32), 4=g_idx(i32), 5=output(f32), 6=params(uniform)
pub fn generate_int4_gemm_gptq_shader() -> String {
    format!(
        r#"// INT4 GEMM GPTQ: input [M,K] x dequant(qweight [K,N]) -> output [M,N]

fn unpack_int4_seq(packed: u32, idx: u32) -> u32 {{
    return (packed >> (idx * 4u)) & 0xFu;
}}

struct GptqParams {{
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> qweight: array<u32>;
@group(0) @binding(2) var<storage, read_write> qzeros: array<u32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> g_idx: array<i32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: GptqParams;

@compute @workgroup_size(16, 16, 1)
fn int4_gemm_gptq(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) {{ return; }}

    let n_packed_zeros = params.n / 8u;

    var acc: f32 = 0.0;
    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {{
        let a = input[row * params.k + ki];
        let pack_row = ki / 8u;
        let sub_idx = ki % 8u;
        let packed = qweight[pack_row * params.n + col];
        let q = unpack_int4_seq(packed, sub_idx);

        let group = u32(g_idx[ki]);
        let zero_pack = qzeros[group * n_packed_zeros + col / 8u];
        let qzero = unpack_int4_seq(zero_pack, col % 8u);

        let scale = scales[group * params.n + col];
        let w = (f32(q) - f32(qzero)) * scale;
        acc = acc + a * w;
    }}
    output[row * params.n + col] = acc;
}}
"#
    )
}

/// Marlin-format INT4 GEMM shader
/// Bindings: 0=input(f32), 1=weight(u32), 2=scales(f32), 3=zeros(f32), 4=output(f32), 5=params(uniform)
pub fn generate_marlin_gemm_shader() -> String {
    format!(
        r#"// Marlin INT4 GEMM: input [M,K] x dequant(weight [K,N]) -> output [M,N]

fn unpack_int4_seq_m(packed: u32, idx: u32) -> u32 {{
    return (packed >> (idx * 4u)) & 0xFu;
}}

struct MarlinParams {{
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> zeros: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: MarlinParams;

@compute @workgroup_size(16, 16, 1)
fn marlin_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) {{ return; }}

    let k_packed = params.k / 8u;

    var acc: f32 = 0.0;
    for (var pack_ki: u32 = 0u; pack_ki < k_packed; pack_ki = pack_ki + 1u) {{
        let packed = weight[pack_ki * params.n + col];
        for (var sub: u32 = 0u; sub < 8u; sub = sub + 1u) {{
            let ki = pack_ki * 8u + sub;
            let a = input[row * params.k + ki];
            let q = unpack_int4_seq_m(packed, sub);
            let group = ki / params.group_size;
            let scale = scales[group * params.n + col];
            let zero = zeros[group * params.n + col];
            let w = (f32(q) - 8.0) * scale + zero;
            acc = acc + a * w;
        }}
    }}
    output[row * params.n + col] = acc;
}}
"#
    )
}
