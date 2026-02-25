//! INT4 per-group quantization (simulated in F32 storage)
//! Two INT4 values packed into one u32 slot.

struct QuantParams {
    num_tokens: u32,
    head_dim: u32,
    group_size: u32,
    mode: u32,
}

// Read-only input (binding 0), read-write packed + scales + zeros (bindings 1-3)
@group(0) @binding(0) var<storage, read> int4_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> int4_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> int4_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> int4_zeros: array<f32>;
@group(0) @binding(4) var<uniform> params: QuantParams;

@compute @workgroup_size(256)
fn quantize_kv_int4_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_idx = gid.x;
    let total_groups = params.num_tokens * params.head_dim / params.group_size;

    if group_idx >= total_groups {
        return;
    }

    let group_start = group_idx * params.group_size;

    var min_val = int4_input[group_start];
    var max_val = int4_input[group_start];

    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let val = int4_input[group_start + i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }

    let scale = (max_val - min_val) / 15.0f;
    let zero = -min_val / max(scale, 1e-8f);

    int4_scales[group_idx] = scale;
    int4_zeros[group_idx] = zero;

    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let val = int4_input[group_start + i];
        let quantized = u32(clamp(i32(round((val - min_val) / max(scale, 1e-8f))), 0, 15));
        let byte_idx = (group_start + i) / 2u;
        let is_high = (group_start + i) % 2u;
        if is_high == 0u {
            int4_packed[byte_idx] = (int4_packed[byte_idx] & 0xF0u) | quantized;
        } else {
            int4_packed[byte_idx] = (int4_packed[byte_idx] & 0x0Fu) | (quantized << 4u);
        }
    }
}
