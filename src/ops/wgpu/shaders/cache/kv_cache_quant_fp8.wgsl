//! FP8-style per-token quantization (simulated in F32 storage)
//! WebGPU has no native FP8/INT8 — quantized values stored as f32.

struct QuantParams {
    num_tokens: u32,
    head_dim: u32,
    group_size: u32,
    mode: u32,
}

// Quantize: read-only input (binding 0), read-write output + scales (bindings 1-2)
@group(0) @binding(0) var<storage, read> quant_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> quant_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> quant_scales: array<f32>;
@group(0) @binding(3) var<uniform> params: QuantParams;

@compute @workgroup_size(256)
fn quantize_kv_fp8_per_token_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= params.num_tokens {
        return;
    }

    var max_val = 0.0f;
    let base = token_idx * params.head_dim;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        max_val = max(max_val, abs(quant_input[base + d]));
    }

    let scale = max(max_val / 127.0f, 1e-8f);
    quant_scales[token_idx] = scale;

    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        let val = quant_input[base + d];
        let quantized = clamp(i32(round(val / scale)), -128, 127);
        quant_output[base + d] = f32(quantized);
    }
}
