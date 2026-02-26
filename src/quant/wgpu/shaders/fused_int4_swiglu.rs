//! WebGPU shader for fused INT4 SwiGLU

/// Fused INT4 dual-GEMM + SwiGLU shader
/// Bindings: 0=input(f32), 1=gate_qw(u32), 2=gate_sc(f32), 3=gate_zr(f32),
///           4=up_qw(u32), 5=up_sc(f32), 6=up_zr(f32), 7=output(f32), 8=params(uniform)
pub fn generate_fused_int4_swiglu_shader() -> String {
    r#"// Fused INT4 SwiGLU: dual GEMM + silu(gate) * up

const SWIGLU_AWQ_SHIFTS = array<u32, 8>(0u, 16u, 4u, 20u, 8u, 24u, 12u, 28u);

fn unpack_awq_swiglu(packed: u32, idx: u32) -> u32 {
    return (packed >> SWIGLU_AWQ_SHIFTS[idx]) & 0xFu;
}

fn silu_f(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

struct SwigluParams {
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> gate_qweight: array<u32>;
@group(0) @binding(2) var<storage, read_write> gate_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate_zeros: array<f32>;
@group(0) @binding(4) var<storage, read_write> up_qweight: array<u32>;
@group(0) @binding(5) var<storage, read_write> up_scales: array<f32>;
@group(0) @binding(6) var<storage, read_write> up_zeros: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@group(0) @binding(8) var<uniform> params: SwigluParams;

@compute @workgroup_size(16, 16, 1)
fn fused_int4_swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) { return; }

    let n_packed = params.n / 8u;
    let pack_col = col / 8u;
    let sub = col % 8u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {
        let a = input[row * params.k + ki];
        let group = ki / params.group_size;

        let gate_packed = gate_qweight[ki * n_packed + pack_col];
        let up_packed = up_qweight[ki * n_packed + pack_col];

        let gq = unpack_awq_swiglu(gate_packed, sub);
        let uq = unpack_awq_swiglu(up_packed, sub);

        let gs = gate_scales[group * params.n + col];
        let gz = gate_zeros[group * params.n + col];
        let us = up_scales[group * params.n + col];
        let uz = up_zeros[group * params.n + col];

        gate_acc = gate_acc + a * (f32(gq) - gz) * gs;
        up_acc = up_acc + a * (f32(uq) - uz) * us;
    }

    output[row * params.n + col] = silu_f(gate_acc) * up_acc;
}
"#
    .to_string()
}
