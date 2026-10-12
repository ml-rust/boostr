//! ALiBi (Attention with Linear Biases) shader
//! Adds position-dependent bias to attention scores in-place
//! bias[i, j] = -slope * |i - j|
//! Slope per head: transformers' ALiBi slopes, see `ops/traits/position/alibi.rs`

struct AlibiParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    position: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: AlibiParams;

/// The ALiBi slope of head h
fn compute_alibi_slope(h: u32, num_heads: u32) -> f32 {
    // transformers / ALiBi paper: P = 2^floor(log2 H).
    var p: u32 = 1u;
    loop {
        if (p * 2u > num_heads) { break; }
        p = p * 2u;
    }
    if (h < p) {
        return pow(2.0, -8.0 * f32(h + 1u) / f32(p));
    }
    return pow(2.0, -4.0 * f32(2u * (h - p) + 1u) / f32(p));
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

@compute @workgroup_size(256)
fn alibi_add_bias_causal_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let total_elems = params.batch_size * params.num_heads * params.seq_len_q * params.seq_len_k;

    if elem_idx >= total_elems {
        return;
    }

    let j = elem_idx % params.seq_len_k;
    let remainder = elem_idx / params.seq_len_k;

    let i = remainder % params.seq_len_q;
    let remainder2 = remainder / params.seq_len_q;

    let h = remainder2 % params.num_heads;

    let abs_q_pos = i + params.position;

    // Causal: future tokens get -inf
    if j > abs_q_pos {
        scores[elem_idx] = -3.402823e+38; // -FLT_MAX (WGSL has no -inf literal)
        return;
    }

    let slope = compute_alibi_slope(h, params.num_heads);
    let distance = f32(abs_q_pos - j);
    scores[elem_idx] += -slope * distance;
}
