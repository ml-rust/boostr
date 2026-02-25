//! Rotary Position Embedding (RoPE) shader
//! Input: x [B, H, S, D], cos_cache [S, D/2], sin_cache [S, D/2]
//! Output: x_rotated [B, H, S, D]
//! Each thread handles one dimension pair (2 elements)

struct RoPEParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,  // Full head dimension (must be even)
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: RoPEParams;

@compute @workgroup_size(256)
fn rope_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let total_pairs = params.batch_size * params.num_heads * params.seq_len * (params.head_dim / 2u);

    if pair_idx >= total_pairs {
        return;
    }

    // Decode linear index to (b, h, s, pair_idx_in_head)
    let half_d = params.head_dim / 2u;
    let pair_in_head = pair_idx % half_d;
    let remainder = pair_idx / half_d;

    let s = remainder % params.seq_len;
    let remainder2 = remainder / params.seq_len;

    let h = remainder2 % params.num_heads;
    let b = remainder2 / params.num_heads;

    // Read cos/sin from cache [s, pair_in_head]
    let cos_idx = s * half_d + pair_in_head;
    let sin_idx = s * half_d + pair_in_head;
    let cos_val = cos_cache[cos_idx];
    let sin_val = sin_cache[sin_idx];

    // Read x[b, h, s, 2*pair_in_head] and x[b, h, s, 2*pair_in_head + 1]
    let x_idx_base = ((b * params.num_heads + h) * params.seq_len + s) * params.head_dim;
    let x1_idx = x_idx_base + 2u * pair_in_head;
    let x2_idx = x_idx_base + 2u * pair_in_head + 1u;

    let x1 = x[x1_idx];
    let x2 = x[x2_idx];

    // Apply rotation: [x1, x2] * [[cos, -sin], [sin, cos]]
    let rot_x1 = x1 * cos_val - x2 * sin_val;
    let rot_x2 = x1 * sin_val + x2 * cos_val;

    // Write output
    out[x1_idx] = rot_x1;
    out[x2_idx] = rot_x2;
}
