//! Rotary Position Embedding (RoPE) shader — split-half pairing
//! Input: x [B, H, S, D], cos_cache [S, D/2], sin_cache [S, D/2]
//! Output: x_rotated [B, H, S, D]
//! Pairs first-half with second-half: (x[d], x[d + D/2]) for d in [0, D/2)
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
    let half_d = params.head_dim / 2u;
    let total_pairs = params.batch_size * params.num_heads * params.seq_len * half_d;

    if pair_idx >= total_pairs {
        return;
    }

    // Decode linear index to (b, h, s, d_pair)
    let d_pair = pair_idx % half_d;
    let remainder = pair_idx / half_d;

    let s = remainder % params.seq_len;
    let remainder2 = remainder / params.seq_len;

    let h = remainder2 % params.num_heads;
    let b = remainder2 / params.num_heads;

    // Read cos/sin from cache [s, d_pair]
    let cache_idx = s * half_d + d_pair;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];

    // Split-half pairing: first half pairs with second half
    let x_base = ((b * params.num_heads + h) * params.seq_len + s) * params.head_dim;
    let idx_first = x_base + d_pair;
    let idx_second = x_base + half_d + d_pair;

    let x_first = x[idx_first];
    let x_second = x[idx_second];

    // Apply rotation
    out[idx_first]  = x_first * cos_val - x_second * sin_val;
    out[idx_second] = x_first * sin_val + x_second * cos_val;
}
