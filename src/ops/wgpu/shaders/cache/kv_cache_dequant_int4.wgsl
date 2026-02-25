//! INT4 per-group dequantization

struct QuantParams {
    num_tokens: u32,
    head_dim: u32,
    group_size: u32,
    mode: u32,
}

// Read-only packed + scales + zeros (bindings 0-2), read-write output (binding 3)
@group(0) @binding(0) var<storage, read> dq_packed: array<u32>;
@group(0) @binding(1) var<storage, read> dq_scales: array<f32>;
@group(0) @binding(2) var<storage, read> dq_zeros: array<f32>;
@group(0) @binding(3) var<storage, read_write> dq_output: array<f32>;
@group(0) @binding(4) var<uniform> params: QuantParams;

@compute @workgroup_size(256)
fn dequantize_kv_int4_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_idx = gid.x;
    let total_groups = params.num_tokens * params.head_dim / params.group_size;

    if group_idx >= total_groups {
        return;
    }

    let scale = dq_scales[group_idx];
    let zero = dq_zeros[group_idx];
    let group_start = group_idx * params.group_size;

    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let byte_idx = (group_start + i) / 2u;
        let is_high = (group_start + i) % 2u;
        let byte_val = dq_packed[byte_idx];
        var quantized: u32;
        if is_high != 0u {
            quantized = (byte_val >> 4u) & 0xFu;
        } else {
            quantized = byte_val & 0xFu;
        }
        dq_output[group_start + i] = f32(quantized) * scale - zero * scale;
    }
}
