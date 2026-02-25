//! Flash Attention v2 - Fused Attention Kernel
//! Q [B, num_heads, S_q, head_dim]
//! K [B, num_kv_heads, S_k, head_dim]
//! V [B, num_kv_heads, S_k, head_dim]
//! Output [B, num_heads, S_q, head_dim]
//! LSE [B, num_heads, S_q] (logsumexp for backward)
//!
//! O(N²) GPU implementation. Handles GQA internally.

struct FlashParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
    window_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> lse: array<f32>;
@group(0) @binding(5) var<uniform> params: FlashParams;

@compute @workgroup_size(256)
fn flash_attention_fwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    let total_queries = params.batch_size * params.num_heads * params.seq_len_q;

    if query_idx >= total_queries {
        return;
    }

    // Decode query position
    let i = query_idx % params.seq_len_q;
    let remainder = query_idx / params.seq_len_q;
    let h_q = remainder % params.num_heads;
    let b = remainder / params.num_heads;

    // GQA: map query head to kv head
    let h_kv = (h_q * params.num_kv_heads) / params.num_heads;

    // Read Q[b, h_q, i, :]
    let q_base = ((b * params.num_heads + h_q) * params.seq_len_q + i) * params.head_dim;

    // Compute attention and aggregate
    var accum: array<f32, 512>;  // Max head_dim = 512
    var max_score = -1e30f;
    var sum_exp = 0.0f;

    // Compute valid key range [start_k, end_k) based on window and causal
    var start_k = 0u;
    var end_k = params.seq_len_k;

    if params.causal != 0u {
        end_k = min(i + 1u, params.seq_len_k);
    }
    if params.window_size > 0u {
        if i >= params.window_size {
            start_k = max(start_k, i - params.window_size + 1u);
        }
    }

    // First pass: find max for stability
    for (var j = start_k; j < end_k; j = j + 1u) {
        let k_base = ((b * params.num_kv_heads + h_kv) * params.seq_len_k + j) * params.head_dim;
        var score = 0.0f;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            score += q[q_base + d] * k[k_base + d];
        }
        score *= params.scale;
        max_score = max(max_score, score);
    }

    // Handle the case where max_score stays at -inf (no valid keys)
    if max_score == -1e30f {
        max_score = 0.0f;
    }

    // Second pass: softmax and aggregate
    for (var j = start_k; j < end_k; j = j + 1u) {
        let k_base = ((b * params.num_kv_heads + h_kv) * params.seq_len_k + j) * params.head_dim;
        var score = 0.0f;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            score += q[q_base + d] * k[k_base + d];
        }
        score *= params.scale;

        let weight = exp(score - max_score);
        sum_exp += weight;

        // Accumulate weighted V
        let v_base = ((b * params.num_kv_heads + h_kv) * params.seq_len_k + j) * params.head_dim;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            accum[d] += weight * v[v_base + d];
        }
    }

    // Normalize
    let out_base = ((b * params.num_heads + h_q) * params.seq_len_q + i) * params.head_dim;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        out[out_base + d] = accum[d] / max(sum_exp, 1e-10f);
    }

    // Store logsumexp: log(sum_exp) + max_score
    let lse_idx = (b * params.num_heads + h_q) * params.seq_len_q + i;
    if sum_exp > 0.0f {
        lse[lse_idx] = log(sum_exp) + max_score;
    } else {
        lse[lse_idx] = -1e30f;  // No valid keys, use -inf
    }
}

// Placeholder backward pass (simplified, real implementation would need gradients)
@compute @workgroup_size(256)
fn flash_attention_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Backward pass would require:
    // - dout, q, k, v, output, lse as inputs
    // - Compute dq, dk, dv
    // For now, this is a placeholder. Full implementation would mirror forward pass
    // with gradient accumulation logic.
}
