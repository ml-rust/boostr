//! INT8 per-token dequantization (simulated in F32 storage)

struct QuantParams {
    num_tokens: u32,
    head_dim: u32,
    group_size: u32,
    mode: u32,
}

// Read-only input + scales (bindings 0-1), read-write output (binding 2)
@group(0) @binding(0) var<storage, read> dq_int8_input: array<f32>;
@group(0) @binding(1) var<storage, read> dq_int8_scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> dq_int8_output: array<f32>;
@group(0) @binding(3) var<uniform> params: QuantParams;

@compute @workgroup_size(256)
fn dequantize_kv_int8_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= params.num_tokens {
        return;
    }

    let scale = dq_int8_scales[token_idx];
    let base = token_idx * params.head_dim;

    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        dq_int8_output[base + d] = dq_int8_input[base + d] * scale;
    }
}
