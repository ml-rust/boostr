//! ALiBi (Attention with Linear Biases) shader
//! Adds position-dependent bias to attention scores in-place
//! bias[i, j] = -slope * |i - j|
//! Slope per head: slope_h = 2^(-8h/H)

struct AlibiParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: AlibiParams;

/// Compute 2^(-8h/H)
fn compute_alibi_slope(h: u32, num_heads: u32) -> f32 {
    let slope_log2 = -8.0 * f32(h) / f32(num_heads);
    return pow(2.0, slope_log2);
}

@compute @workgroup_size(256)
fn alibi_add_bias_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let total_elems = params.batch_size * params.num_heads * params.seq_len_q * params.seq_len_k;

    if elem_idx >= total_elems {
        return;
    }

    // Decode index to (b, h, i, j)
    let j = elem_idx % params.seq_len_k;
    let remainder = elem_idx / params.seq_len_k;

    let i = remainder % params.seq_len_q;
    let remainder2 = remainder / params.seq_len_q;

    let h = remainder2 % params.num_heads;
    let _b = remainder2 / params.num_heads;

    // Compute bias: -slope * |i - j|
    let slope = compute_alibi_slope(h, params.num_heads);
    let distance = f32(i) - f32(j);
    let bias = -slope * abs(distance);

    // Add bias in-place
    scores[elem_idx] += bias;
}
