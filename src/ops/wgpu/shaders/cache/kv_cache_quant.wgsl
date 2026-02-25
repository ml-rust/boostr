//! KV cache quantization kernels
//! FP8, INT4, INT8 quantization and dequantization

struct QuantParams {
    num_tokens: u32,
    head_dim: u32,
    group_size: u32,  // For INT4
    mode: u32,        // 0=per_tensor, 1=per_token
}

// ============ FP8 Per-Token Quantization ============

@group(0) @binding(0) var<storage, read> input_fp32: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_fp8: array<u32>;  // Packed as u32
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: QuantParams;

@compute @workgroup_size(256)
fn quantize_kv_fp8_per_token_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= params.num_tokens {
        return;
    }

    // Find max absolute value in this token across all dimensions
    var max_val = 0.0f;
    let base = token_idx * params.head_dim;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        max_val = max(max_val, abs(input_fp32[base + d]));
    }

    // Compute scale: max_val / 127.0 (E4M3 range)
    let scale = max(max_val / 127.0f, 1e-8f);
    scales[token_idx] = scale;

    // Quantize to FP8
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        let val = input_fp32[base + d];
        let quantized = i32(round(val / scale));
        let clamped = clamp(quantized, -128, 127);
        output_fp8[base + d] = bitcast<u32>(u8(clamped));
    }
}

// ============ FP8 Per-Token Dequantization ============

@group(0) @binding(0) var<storage, read> input_fp8: array<u32>;
@group(0) @binding(1) var<storage, read> quant_scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_fp32: array<f32>;
@group(0) @binding(3) var<uniform> dequant_params: QuantParams;

@compute @workgroup_size(256)
fn dequantize_kv_fp8_per_token_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= dequant_params.num_tokens {
        return;
    }

    let scale = quant_scales[token_idx];
    let base = token_idx * dequant_params.head_dim;

    for (var d = 0u; d < dequant_params.head_dim; d = d + 1u) {
        let quantized_val = i8(bitcast<i32>(input_fp8[base + d]));
        output_fp32[base + d] = f32(quantized_val) * scale;
    }
}

// ============ INT4 Per-Group Quantization ============

@compute @workgroup_size(256)
fn quantize_kv_int4_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_idx = gid.x;
    let total_groups = params.num_tokens * params.head_dim / params.group_size;

    if group_idx >= total_groups {
        return;
    }

    let group_start = group_idx * params.group_size;

    // Find min/max in this group
    var min_val = input_fp32[group_start];
    var max_val = input_fp32[group_start];

    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let val = input_fp32[group_start + i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }

    // Compute asymmetric scale and zero
    let scale = (max_val - min_val) / 15.0f;  // INT4 range: 0-15
    let zero = -min_val / max(scale, 1e-8f);

    scales[group_idx] = scale;
    scales[params.num_tokens + group_idx] = zero;  // Store zero after scales

    // Quantize to INT4
    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let val = input_fp32[group_start + i];
        let quantized = clamp(i32(round((val - min_val) / max(scale, 1e-8f))), 0, 15);
        // Pack two INT4 values per byte
        let byte_idx = (group_start + i) / 2u;
        let is_high = (group_start + i) % 2u;
        if is_high == 0u {
            output_fp8[byte_idx] = (output_fp8[byte_idx] & 0xF0u) | u32(quantized);
        } else {
            output_fp8[byte_idx] = (output_fp8[byte_idx] & 0x0Fu) | (u32(quantized) << 4u);
        }
    }
}

// ============ INT4 Per-Group Dequantization ============

@compute @workgroup_size(256)
fn dequantize_kv_int4_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_idx = gid.x;
    let total_groups = dequant_params.num_tokens * dequant_params.head_dim / dequant_params.group_size;

    if group_idx >= total_groups {
        return;
    }

    let scale = quant_scales[group_idx];
    let zero = quant_scales[dequant_params.num_tokens + group_idx];
    let group_start = group_idx * dequant_params.group_size;

    for (var i = 0u; i < dequant_params.group_size; i = i + 1u) {
        let byte_idx = (group_start + i) / 2u;
        let is_high = (group_start + i) % 2u;
        let byte_val = input_fp8[byte_idx];
        let quantized = if is_high != 0u { (byte_val >> 4u) & 0xFu } else { byte_val & 0xFu };
        output_fp32[group_start + i] = f32(quantized) * scale - zero * scale;
    }
}

// ============ INT8 Per-Token Quantization ============

@compute @workgroup_size(256)
fn quantize_kv_int8_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= params.num_tokens {
        return;
    }

    // Find max absolute value
    var max_val = 0.0f;
    let base = token_idx * params.head_dim;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        max_val = max(max_val, abs(input_fp32[base + d]));
    }

    // Compute scale
    let scale = max(max_val / 127.0f, 1e-8f);
    scales[token_idx] = scale;

    // Quantize to INT8
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        let val = input_fp32[base + d];
        let quantized = i32(round(val / scale));
        let clamped = clamp(quantized, -128, 127);
        output_fp8[base + d] = bitcast<u32>(u8(clamped));
    }
}

// ============ INT8 Per-Token Dequantization ============

@compute @workgroup_size(256)
fn dequantize_kv_int8_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= dequant_params.num_tokens {
        return;
    }

    let scale = quant_scales[token_idx];
    let base = token_idx * dequant_params.head_dim;

    for (var d = 0u; d < dequant_params.head_dim; d = d + 1u) {
        let quantized_val = i8(bitcast<i32>(input_fp8[base + d]));
        output_fp32[base + d] = f32(quantized_val) * scale;
    }
}
