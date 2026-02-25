//! Scaled Dot-Product Attention (SDPA) for MLA
//! Supports different K and V last dimensions
//! Q [B, H, S_q, D_k], K [B, H, S_k, D_k], V [B, H, S_k, D_v]
//! Output [B, H, S_q, D_v]
//!
//! O(N²) implementation using simple loop structure.
//! Each workgroup handles one (batch, head, query_position) combination.

struct SdpaParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    head_dim_k: u32,
    head_dim_v: u32,
    scale: f32,
    causal: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: SdpaParams;

@compute @workgroup_size(256)
fn sdpa_forward_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    let total_queries = params.batch_size * params.num_heads * params.seq_len_q;

    if query_idx >= total_queries {
        return;
    }

    // Decode query position
    let i = query_idx % params.seq_len_q;
    let remainder = query_idx / params.seq_len_q;
    let h = remainder % params.num_heads;
    let b = remainder / params.num_heads;

    // Read query vector Q[b, h, i, :] (head_dim_k elements)
    let q_base = ((b * params.num_heads + h) * params.seq_len_q + i) * params.head_dim_k;

    // Compute attention scores for all keys and aggregate values
    var accum: array<f32, 512>;  // Max head_dim_v = 512
    var max_score = -1e30f;
    var sum_exp = 0.0f;

    // First pass: find max score for numerical stability
    for (var j = 0u; j < params.seq_len_k; j = j + 1u) {
        if params.causal != 0u && i < j {
            // Causal mask: ignore future positions
            continue;
        }

        // Q @ K^T
        let k_base = ((b * params.num_heads + h) * params.seq_len_k + j) * params.head_dim_k;
        var score = 0.0f;
        for (var d = 0u; d < params.head_dim_k; d = d + 1u) {
            score += q[q_base + d] * k[k_base + d];
        }
        score *= params.scale;
        max_score = max(max_score, score);
    }

    // Second pass: compute softmax and weighted sum
    for (var j = 0u; j < params.seq_len_k; j = j + 1u) {
        if params.causal != 0u && i < j {
            continue;
        }

        let k_base = ((b * params.num_heads + h) * params.seq_len_k + j) * params.head_dim_k;
        var score = 0.0f;
        for (var d = 0u; d < params.head_dim_k; d = d + 1u) {
            score += q[q_base + d] * k[k_base + d];
        }
        score *= params.scale;

        let weight = exp(score - max_score);
        sum_exp += weight;

        // Accumulate weighted V
        let v_base = ((b * params.num_heads + h) * params.seq_len_k + j) * params.head_dim_v;
        for (var d = 0u; d < params.head_dim_v; d = d + 1u) {
            accum[d] += weight * v[v_base + d];
        }
    }

    // Normalize and write output
    let out_base = ((b * params.num_heads + h) * params.seq_len_q + i) * params.head_dim_v;
    for (var d = 0u; d < params.head_dim_v; d = d + 1u) {
        out[out_base + d] = accum[d] / sum_exp;
    }
}
