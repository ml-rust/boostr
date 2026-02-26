//! WebGPU shader for fused INT4 QKV projection

/// Fused INT4 QKV shader — computes Q, K, V as separate kernels (invoked three times)
/// Bindings: 0=input(f32), 1=qw_q(u32), 2=sc_q(f32), 3=zr_q(f32),
///           4=qw_k(u32), 5=sc_k(f32), 6=zr_k(f32),
///           7=qw_v(u32), 8=sc_v(f32), 9=zr_v(f32),
///           10=out_q(f32), 11=out_k(f32), 12=out_v(f32), 13=params(uniform)
pub fn generate_fused_int4_qkv_shader() -> String {
    format!(
        r#"// Fused INT4 QKV: computes Q, K, V projections via separate dispatches

const QKV_AWQ_SHIFTS = array<u32, 8>(0u, 16u, 4u, 20u, 8u, 24u, 12u, 28u);

fn unpack_awq_qkv(packed: u32, idx: u32) -> u32 {{
    return (packed >> QKV_AWQ_SHIFTS[idx]) & 0xFu;
}}

struct QkvParams {{
    m: u32,
    k: u32,
    nq: u32,
    nkv: u32,
    group_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> qw_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> sc_q: array<f32>;
@group(0) @binding(3) var<storage, read_write> zr_q: array<f32>;
@group(0) @binding(4) var<storage, read_write> qw_k: array<u32>;
@group(0) @binding(5) var<storage, read_write> sc_k: array<f32>;
@group(0) @binding(6) var<storage, read_write> zr_k: array<f32>;
@group(0) @binding(7) var<storage, read_write> qw_v: array<u32>;
@group(0) @binding(8) var<storage, read_write> sc_v: array<f32>;
@group(0) @binding(9) var<storage, read_write> zr_v: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_k: array<f32>;
@group(0) @binding(12) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(13) var<uniform> params: QkvParams;

// Dispatch Q projection
@compute @workgroup_size(16, 16, 1)
fn fused_int4_qkv_q(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.nq) {{ return; }}

    let n_packed = params.nq / 8u;
    let pack_col = col / 8u;
    let sub = col % 8u;

    var acc: f32 = 0.0;
    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {{
        let a = input[row * params.k + ki];
        let packed = qw_q[ki * n_packed + pack_col];
        let q = unpack_awq_qkv(packed, sub);
        let group = ki / params.group_size;
        let w = (f32(q) - zr_q[group * params.nq + col]) * sc_q[group * params.nq + col];
        acc = acc + a * w;
    }}
    out_q[row * params.nq + col] = acc;
}}

// Dispatch K projection
@compute @workgroup_size(16, 16, 1)
fn fused_int4_qkv_k(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.nkv) {{ return; }}

    let n_packed = params.nkv / 8u;
    let pack_col = col / 8u;
    let sub = col % 8u;

    var acc: f32 = 0.0;
    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {{
        let a = input[row * params.k + ki];
        let packed = qw_k[ki * n_packed + pack_col];
        let q = unpack_awq_qkv(packed, sub);
        let group = ki / params.group_size;
        let w = (f32(q) - zr_k[group * params.nkv + col]) * sc_k[group * params.nkv + col];
        acc = acc + a * w;
    }}
    out_k[row * params.nkv + col] = acc;
}}

// Dispatch V projection
@compute @workgroup_size(16, 16, 1)
fn fused_int4_qkv_v(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.nkv) {{ return; }}

    let n_packed = params.nkv / 8u;
    let pack_col = col / 8u;
    let sub = col % 8u;

    var acc: f32 = 0.0;
    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {{
        let a = input[row * params.k + ki];
        let packed = qw_v[ki * n_packed + pack_col];
        let q = unpack_awq_qkv(packed, sub);
        let group = ki / params.group_size;
        let w = (f32(q) - zr_v[group * params.nkv + col]) * sc_v[group * params.nkv + col];
        acc = acc + a * w;
    }}
    out_v[row * params.nkv + col] = acc;
}}
"#
    )
}
